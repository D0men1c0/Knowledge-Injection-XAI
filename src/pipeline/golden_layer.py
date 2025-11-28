"""
Gold Layer: Meta-Learning & Insight Generation.

Responsibilities:
1. Distributed aggregation of XAI metrics per adapter.
2. Centralized meta-learning for error prediction.
3. Distributed generation of final meta-dataset via Pandas UDF.
"""

from __future__ import annotations

from typing import Iterator
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType
)
from pyspark.sql import functions as F

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from src.configs import ExperimentConfig
from src.utils.telemetry import logger


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
def get_meta_schema() -> StructType:
    return StructType([
        StructField("image_path", StringType(), False),
        StructField("entropy", FloatType(), False),
        StructField("sparsity", FloatType(), False),
        StructField("deletion_score", FloatType(), False),
        StructField("insertion_score", FloatType(), False),
        StructField("is_correct", IntegerType(), False),
        StructField("meta_probability", FloatType(), False),
        StructField("meta_prediction", IntegerType(), False),
    ])


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def load_silver_data(spark: SparkSession, cfg: ExperimentConfig) -> DataFrame:
    logger.info(f"Loading Silver data from: {cfg.paths.silver}")

    df = (
        spark.read.parquet(cfg.paths.silver)
        .withColumn("pred_clean", F.regexp_replace(col("predicted_class"), "LABEL_", ""))
        .withColumn("is_correct", F.when(col("pred_clean") == col("true_label"), 1).otherwise(0))
        .withColumn("deletion_score", col("deletion_score").cast(FloatType()))
        .withColumn("entropy", col("entropy").cast(FloatType()))
        .withColumn("sparsity", col("sparsity").cast(FloatType()))
        .select(
            "image_path",
            "adapter_rank",
            "entropy",
            "sparsity",
            "deletion_score",
            "insertion_score",
            "is_correct",
        )
        .repartition(8)
        .persist()
    )

    logger.info(f"Silver rows loaded: {df.count()}")
    return df


# ---------------------------------------------------------------------
# Aggregation (Spark)
# ---------------------------------------------------------------------
def aggregate_metrics_by_adapter(df: DataFrame, cfg: ExperimentConfig) -> None:
    logger.info("Running adapter-level aggregation")

    agg_df = (
        df.groupBy("adapter_rank")
          .agg(
              F.mean("is_correct").alias("accuracy"),
              F.mean("deletion_score").alias("avg_deletion_score"),
              F.mean("entropy").alias("avg_entropy"),
              F.mean("sparsity").alias("avg_sparsity"),
              F.count("image_path").alias("sample_count"),
          )
          .orderBy("adapter_rank")
    )

    output_path = Path(cfg.paths.gold) / "adapter_ranking.parquet"
    agg_df.write.mode("overwrite").parquet(str(output_path))

    logger.info(f"Adapter aggregation saved to {output_path}")


# ---------------------------------------------------------------------
# Meta-Learner (Driver Only)
# ---------------------------------------------------------------------
def train_meta_learner(df: DataFrame, plots_dir: Path, cfg: ExperimentConfig):
    logger.info("Collecting meta dataset to driver for training")

    pdf = df.select(
        "entropy",
        "sparsity",
        "deletion_score",
        "insertion_score",
        "is_correct",
    ).toPandas().dropna()

    if pdf.empty:
        logger.warning("No valid rows for meta-learning")
        return None

    X = pdf[["entropy", "sparsity", "deletion_score", "insertion_score"]]
    y = pdf["is_correct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=2,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )

    stacking_clf = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
        n_jobs=-1,
    )

    logger.info("Training stacking meta-learner")
    stacking_clf.fit(X_train, y_train)

    # Evaluation
    y_pred = stacking_clf.predict(X_test)
    y_proba = stacking_clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    logger.info(f"Meta-Learner Report:\n{report}")
    logger.info(f"Meta-Learner ROC-AUC: {auc:.4f}")

    # Save model
    model_path = Path(cfg.paths.gold) / "meta_learner.joblib"
    joblib.dump(stacking_clf, model_path)
    logger.info(f"Meta-learner saved to {model_path}")

    plot_feature_importance(rf_model, X.columns, plots_dir / "rf_feature_importance.png")
    plot_confusion_matrix(y_test, y_pred, plots_dir / "meta_confusion_matrix.png")

    return model_path

_model = None
def load_model_once(model_path: Path):
    global _model
    if _model is None:
        _model = joblib.load(model_path)
    return _model

# ---------------------------------------------------------------------
# Distributed Meta Inference (UDF)
# ---------------------------------------------------------------------
def run_meta_inference(df: DataFrame, model_path: Path, cfg: ExperimentConfig):

    bc_model_path = df.sql_ctx.sparkSession.sparkContext.broadcast(str(model_path))

    @pandas_udf(get_meta_schema())
    def meta_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        model = load_model_once(bc_model_path.value)

        for batch in iterator:
            X = batch[[
                "entropy",
                "sparsity",
                "deletion_score",
                "insertion_score",
            ]]

            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X)

            batch["meta_probability"] = proba.astype("float32")
            batch["meta_prediction"] = pred.astype("int32")

            yield batch


    meta_df = (
        df.repartition(8)
          .select(
              "image_path",
              "entropy",
              "sparsity",
              "deletion_score",
              "insertion_score",
              "is_correct",
          )
          .select(meta_udf(F.struct("*")).alias("data"))
          .select("data.*")
    )

    output_path = Path(cfg.paths.gold) / "meta_insights.parquet"
    meta_df.write.mode("overwrite").parquet(str(output_path))

    logger.info(f"Distributed meta-insight dataset saved to {output_path}")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_feature_importance(model, features, output_path: Path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=[features[i] for i in indices],
    )
    plt.title("XAI Metric Importance for Error Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Error (0)", "Correct (1)"],
        yticklabels=["Error (0)", "Correct (1)"],
    )
    plt.ylabel("Ground Truth")
    plt.xlabel("Meta-Learner Prediction")
    plt.title("Meta-Learner Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Confusion matrix plot saved to {output_path}")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
def run_gold(spark: SparkSession, cfg: ExperimentConfig) -> None:
    logger.info("Starting Gold Layer")

    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.gold).mkdir(parents=True, exist_ok=True)

    # Load + Cache
    df_silver = load_silver_data(spark, cfg)

    # Distributed Aggregation
    aggregate_metrics_by_adapter(df_silver, cfg)

    # Train Meta Learner (Driver)
    model_path = train_meta_learner(df_silver, plots_dir, cfg)

    # Distributed Meta Inference (UDF)
    if model_path is not None:
        run_meta_inference(df_silver, model_path, cfg)

    df_silver.unpersist()
    logger.info("Gold Layer completed successfully")