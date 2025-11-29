"""
Gold Layer: XAI-Robustness Correlation Analysis.

Validates hypothesis: XAI metrics on clean data predict OOD robustness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import joblib

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr

from src.configs import ExperimentConfig
from src.utils.telemetry import logger


FEATURE_COLS = ["entropy", "sparsity", "deletion_score", "insertion_score"]


def load_and_join_data(spark: SparkSession, cfg: ExperimentConfig) -> DataFrame:
    """Join Silver (XAI) with OOD (is_correct) using broadcast join."""
    df_silver = (
        spark.read.parquet(cfg.paths.silver)
        .select("image_path", "adapter_rank", *FEATURE_COLS)
    )
    df_ood = (
        spark.read.parquet(cfg.paths.ood)
        .select("image_path", "adapter_rank", "corruption_type", "is_correct")
    )

    df_joined = df_silver.join(
        df_ood,
        on=["image_path", "adapter_rank"],
        how="inner",
    ).cache()

    logger.info(f"Joined dataset: {df_joined.count()} rows")
    return df_joined


def compute_correlations(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson/Spearman correlations: XAI features vs is_correct."""
    y = pdf["is_correct"].values
    results = []

    for col in FEATURE_COLS:
        x = pdf[col].values
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        results.append({
            "feature": col,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
        })
        logger.info(f"{col}: Pearson={pr:.4f}, Spearman={sr:.4f}")

    return pd.DataFrame(results)


def train_robustness_classifier(
    pdf: pd.DataFrame, output_path: Path
) -> Tuple[str, pd.DataFrame]:
    """Train RandomForest: XAI features → is_correct."""
    X = pdf[FEATURE_COLS].values
    y = pdf["is_correct"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Classifier: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    logger.info(f"Feature Importance:\n{importance_df.to_string(index=False)}")

    model_path = output_path / "robustness_classifier.joblib"
    joblib.dump(clf, model_path)

    return str(model_path), importance_df


def compute_aggregated_stats(df: DataFrame) -> DataFrame:
    """Aggregated stats using Spark-native operations."""
    return (
        df.groupBy("adapter_rank", "corruption_type")
        .agg(
            F.mean("entropy").alias("mean_entropy"),
            F.mean("sparsity").alias("mean_sparsity"),
            F.mean("deletion_score").alias("mean_deletion"),
            F.mean("insertion_score").alias("mean_insertion"),
            F.mean("is_correct").alias("ood_accuracy"),
            F.sum("is_correct").alias("correct_count"),
            F.count("*").alias("total_count"),
        )
    )


def run_gold(spark: SparkSession, cfg: ExperimentConfig) -> None:
    """Execute Gold Layer pipeline."""
    logger.info("Starting Gold Layer")

    gold_path = Path(cfg.paths.gold)
    gold_path.mkdir(parents=True, exist_ok=True)

    df_joined = load_and_join_data(spark, cfg)
    df_joined.write.mode("overwrite").parquet(str(gold_path / "joined_dataset.parquet"))

    # Aggregated stats (Spark-native)
    agg_df = compute_aggregated_stats(df_joined)
    agg_df.write.mode("overwrite").parquet(str(gold_path / "aggregated_stats.parquet"))
    agg_df.show(truncate=False)

    # Pandas for ML
    pdf = df_joined.toPandas()

    corr_df = compute_correlations(pdf)
    corr_df.to_parquet(gold_path / "correlations.parquet", index=False)

    _, importance_df = train_robustness_classifier(pdf, gold_path)
    importance_df.to_parquet(gold_path / "feature_importance.parquet", index=False)

    df_joined.unpersist()
    logger.info("Gold Layer completed")