"""
Gold Layer: XAI-Robustness Correlation Analysis.

Full Spark optimization with Arrow, broadcast joins, window functions.
Validates hypothesis: XAI metrics on clean data predict OOD robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import joblib

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.configs import ExperimentConfig
from src.utils.telemetry import logger


FEATURE_COLS = ["entropy", "sparsity", "deletion_score", "insertion_score"]


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass
class MetricResult:
    """Single metric result container."""
    name: str
    value: float
    category: str  # "qualitative" | "quantitative"


# =============================================================================
# SPARK-NATIVE AGGREGATIONS
# =============================================================================
def load_and_join(spark: SparkSession, cfg: ExperimentConfig) -> DataFrame:
    """Join Silver + OOD with broadcast optimization."""
    df_silver = (
        spark.read.parquet(cfg.paths.silver)
        .select("image_path", "adapter_rank", *FEATURE_COLS)
    )
    df_ood = (
        spark.read.parquet(cfg.paths.ood)
        .select("image_path", "adapter_rank", "corruption_type", "corruption_level", "is_correct")
    )

    silver_count = df_silver.count()
    ood_count = df_ood.count()

    if silver_count < ood_count * 0.3:
        df = df_ood.join(F.broadcast(df_silver), ["image_path", "adapter_rank"], "inner")
    else:
        df = df_silver.join(df_ood, ["image_path", "adapter_rank"], "inner")

    return df.repartition(8, "adapter_rank", "corruption_type").cache()


def compute_base_stats(df: DataFrame) -> DataFrame:
    """Core aggregations: single Spark pass."""
    return (
        df.groupBy("adapter_rank", "corruption_type", "corruption_level")
        .agg(
            F.mean("entropy").alias("mean_entropy"),
            F.mean("sparsity").alias("mean_sparsity"),
            F.mean("deletion_score").alias("mean_deletion"),
            F.mean("insertion_score").alias("mean_insertion"),
            F.stddev("entropy").alias("std_entropy"),
            F.stddev("sparsity").alias("std_sparsity"),
            F.stddev("deletion_score").alias("std_deletion"),
            F.stddev("insertion_score").alias("std_insertion"),
            F.mean("is_correct").alias("accuracy"),
            F.sum("is_correct").alias("correct_count"),
            F.count("*").alias("total_count"),
            F.variance("entropy").alias("var_entropy"),
        )
        .withColumn("error_rate", 1.0 - F.col("accuracy"))
    )


def compute_degradation(df: DataFrame) -> DataFrame:
    """Accuracy degradation shallow→heavy via window functions."""
    level_order = (
        F.when(F.col("corruption_level") == "shallow", 0)
        .when(F.col("corruption_level") == "medium", 1)
        .otherwise(2)
    )

    agg = (
        df.groupBy("adapter_rank", "corruption_type", "corruption_level")
        .agg(F.mean("is_correct").alias("accuracy"))
        .withColumn("level_order", level_order)
    )

    pivot = (
        agg.groupBy("adapter_rank", "corruption_type")
        .pivot("corruption_level", ["shallow", "medium", "heavy"])
        .agg(F.first("accuracy"))
    )

    return (
        pivot
        .withColumn("drop_shallow_heavy", F.col("shallow") - F.col("heavy"))
        .withColumn("drop_pct", (F.col("shallow") - F.col("heavy")) / F.col("shallow") * 100)
        .withColumn("drop_shallow_medium", F.col("shallow") - F.col("medium"))
        .withColumn("drop_medium_heavy", F.col("medium") - F.col("heavy"))
    )


def compute_xai_global(df: DataFrame) -> DataFrame:
    """Global XAI summary."""
    return df.select(
        *[F.mean(c).alias(f"mean_{c}") for c in FEATURE_COLS],
        *[F.stddev(c).alias(f"std_{c}") for c in FEATURE_COLS],
        *[F.min(c).alias(f"min_{c}") for c in FEATURE_COLS],
        *[F.max(c).alias(f"max_{c}") for c in FEATURE_COLS],
        *[F.expr(f"percentile_approx({c}, 0.5)").alias(f"median_{c}") for c in FEATURE_COLS],
        *[F.expr(f"percentile_approx({c}, 0.25)").alias(f"q25_{c}") for c in FEATURE_COLS],
        *[F.expr(f"percentile_approx({c}, 0.75)").alias(f"q75_{c}") for c in FEATURE_COLS],
    )


def compute_adapter_summary(df: DataFrame) -> DataFrame:
    """Per-adapter performance."""
    return (
        df.groupBy("adapter_rank")
        .agg(
            F.mean("is_correct").alias("accuracy"),
            F.mean("entropy").alias("mean_entropy"),
            F.mean("sparsity").alias("mean_sparsity"),
            F.mean("deletion_score").alias("mean_deletion"),
            F.mean("insertion_score").alias("mean_insertion"),
            F.stddev("entropy").alias("std_entropy"),
            F.count("*").alias("n_samples"),
        )
        .withColumn("rank_int", F.col("adapter_rank").cast("int"))
        .orderBy("rank_int")
        .drop("rank_int")
    )


def compute_corruption_impact(df: DataFrame) -> DataFrame:
    """Rank corruptions by impact (worst accuracy first)."""
    agg = df.groupBy("adapter_rank", "corruption_type").agg(
        F.mean("is_correct").alias("accuracy")
    )
    window = Window.partitionBy("adapter_rank").orderBy("accuracy")
    return agg.withColumn("impact_rank", F.row_number().over(window))


def compute_feature_separation(df: DataFrame) -> DataFrame:
    """Feature separation correct vs wrong."""
    correct = df.filter(F.col("is_correct") == 1)
    wrong = df.filter(F.col("is_correct") == 0)

    c_stats = correct.select(
        *[F.mean(c).alias(f"{c}_correct_mean") for c in FEATURE_COLS],
        *[F.stddev(c).alias(f"{c}_correct_std") for c in FEATURE_COLS],
    )
    w_stats = wrong.select(
        *[F.mean(c).alias(f"{c}_wrong_mean") for c in FEATURE_COLS],
        *[F.stddev(c).alias(f"{c}_wrong_std") for c in FEATURE_COLS],
    )
    return c_stats.crossJoin(w_stats)


def compute_xai_by_corruption(df: DataFrame) -> DataFrame:
    """XAI metrics per corruption type/level."""
    return (
        df.groupBy("corruption_type", "corruption_level")
        .agg(
            F.mean("entropy").alias("mean_entropy"),
            F.mean("sparsity").alias("mean_sparsity"),
            F.mean("deletion_score").alias("mean_deletion"),
            F.mean("insertion_score").alias("mean_insertion"),
            F.mean("is_correct").alias("accuracy"),
            F.count("*").alias("n_samples"),
        )
    )


def compute_worst_cases(df: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Identify low-accuracy scenarios (potential failure modes)."""
    agg = (
        df.groupBy("adapter_rank", "corruption_type", "corruption_level")
        .agg(F.mean("is_correct").alias("accuracy"))
    )
    return agg.filter(F.col("accuracy") < threshold).orderBy("accuracy")


# =============================================================================
# PANDAS ANALYSIS (Arrow transfer)
# =============================================================================
def compute_correlations(pdf: pd.DataFrame) -> pd.DataFrame:
    """Correlation metrics (Pandas for scipy stats)."""
    from scipy.stats import pearsonr, spearmanr, pointbiserialr

    y = pdf["is_correct"].values
    results = []

    for col in FEATURE_COLS:
        x = pdf[col].values
        x_c, x_w = x[y == 1], x[y == 0]

        pr, pp = pearsonr(x, y) if len(x) > 2 else (0, 1)
        sr, sp = spearmanr(x, y) if len(x) > 2 else (0, 1)
        pb, pbp = pointbiserialr(x, y) if len(x) > 2 else (0, 1)

        pooled = np.sqrt(
            ((len(x_c)-1)*x_c.std()**2 + (len(x_w)-1)*x_w.std()**2) /
            max(len(x_c) + len(x_w) - 2, 1)
        )
        cohens_d = (x_c.mean() - x_w.mean()) / pooled if pooled > 0 else 0
        sep = abs(x_c.mean() - x_w.mean()) / (x_c.std() + x_w.std() + 1e-8)

        results.append({
            "feature": col,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "point_biserial_r": pb,
            "cohens_d": cohens_d,
            "separation_ratio": sep,
            "mean_correct": x_c.mean(),
            "mean_wrong": x_w.mean(),
            "std_correct": x_c.std(),
            "std_wrong": x_w.std(),
        })

    return pd.DataFrame(results)


def train_meta_learner(pdf: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
    """Train meta-learner: XAI → OOD robustness."""
    X = pdf[FEATURE_COLS].values
    y = pdf["is_correct"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        ),
    }

    results = []
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(clf, X_sc, y, cv=cv, scoring='roc_auc')

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_te, y_pred),
            "roc_auc": roc_auc_score(y_te, y_proba),
            "f1": f1_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred),
            "recall": recall_score(y_te, y_pred),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "brier": brier_score_loss(y_te, y_proba),
            "avg_precision": average_precision_score(y_te, y_proba),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std": cv_auc.std(),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })

    # Feature importance
    rf = models["RandomForest"]
    perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1)
    lr = models["LogisticRegression"]

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gini_importance": rf.feature_importances_,
        "perm_importance": perm.importances_mean,
        "perm_std": perm.importances_std,
        "lr_coef": lr.coef_[0],
        "lr_odds_ratio": np.exp(lr.coef_[0]),
    }).sort_values("perm_importance", ascending=False)

    joblib.dump({"model": rf, "scaler": scaler}, output_path / "meta_learner.joblib")

    return {"results": pd.DataFrame(results), "importance": imp}


def build_summary_tables(
    corr_df: pd.DataFrame,
    deg_pdf: pd.DataFrame,
    ml_results: Dict[str, Any],
    adapter_pdf: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build final summary tables for visualization."""

    # Qualitative summary
    qual = pd.DataFrame([
        {"metric": "Best XAI predictor", "value": corr_df.loc[corr_df["cohens_d"].abs().idxmax(), "feature"]},
        {"metric": "Highest correlation", "value": f"{corr_df['pearson_r'].abs().max():.3f}"},
        {"metric": "Best effect size (Cohen's d)", "value": f"{corr_df['cohens_d'].abs().max():.3f}"},
        {"metric": "Best meta-learner", "value": ml_results["results"].loc[ml_results["results"]["roc_auc"].idxmax(), "model"]},
        {"metric": "Meta-learner AUC", "value": f"{ml_results['results']['roc_auc'].max():.3f}"},
    ])

    # Quantitative summary
    quant = pd.DataFrame([
        {"metric": "Adapters tested", "value": str(len(adapter_pdf))},
        {"metric": "Corruption types", "value": str(len(deg_pdf["corruption_type"].unique()))},
        {"metric": "Max accuracy drop (%)", "value": f"{deg_pdf['drop_pct'].max():.2f}"},
        {"metric": "Avg accuracy drop (%)", "value": f"{deg_pdf['drop_pct'].mean():.2f}"},
    ])

    # XAI feature ranking
    xai_rank = ml_results["importance"][["feature", "gini_importance", "perm_importance", "lr_odds_ratio"]].copy()
    xai_rank["rank"] = range(1, len(xai_rank) + 1)

    # Adapter ranking
    adapter_rank = adapter_pdf[["adapter_rank", "accuracy", "mean_entropy", "mean_sparsity"]].copy()
    adapter_rank = adapter_rank.sort_values("accuracy", ascending=False).reset_index(drop=True)
    adapter_rank["rank"] = range(1, len(adapter_rank) + 1)

    # Worst corruption per adapter
    worst = deg_pdf.loc[deg_pdf.groupby("adapter_rank")["drop_pct"].idxmax()][
        ["adapter_rank", "corruption_type", "drop_pct"]
    ].rename(columns={"corruption_type": "worst_corruption", "drop_pct": "max_drop_pct"})

    return {
        "qualitative_summary": qual,
        "quantitative_summary": quant,
        "xai_feature_ranking": xai_rank,
        "adapter_ranking": adapter_rank,
        "worst_corruption_per_adapter": worst,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_gold(spark: SparkSession, cfg: ExperimentConfig) -> None:
    """Execute Gold Layer: Spark-optimized analysis."""
    logger.info("Starting Gold Layer")

    gold = Path(cfg.paths.gold)
    gold.mkdir(parents=True, exist_ok=True)

    # 1. Load + join
    logger.info("Loading data...")
    df = load_and_join(spark, cfg)
    n = df.count()
    logger.info(f"Joined: {n:,} rows")

    # 2. Spark aggregations
    logger.info("Computing Spark aggregations...")

    base_stats = compute_base_stats(df)
    base_stats.write.mode("overwrite").parquet(str(gold / "base_stats.parquet"))

    deg_df = compute_degradation(df)
    deg_df.write.mode("overwrite").parquet(str(gold / "degradation.parquet"))

    xai_global = compute_xai_global(df)
    xai_global.write.mode("overwrite").parquet(str(gold / "xai_global.parquet"))

    adapter_df = compute_adapter_summary(df)
    adapter_df.write.mode("overwrite").parquet(str(gold / "adapter_summary.parquet"))
    adapter_df.show(truncate=False)

    impact_df = compute_corruption_impact(df)
    impact_df.write.mode("overwrite").parquet(str(gold / "corruption_impact.parquet"))

    sep_df = compute_feature_separation(df)
    sep_df.write.mode("overwrite").parquet(str(gold / "feature_separation.parquet"))

    xai_corr = compute_xai_by_corruption(df)
    xai_corr.write.mode("overwrite").parquet(str(gold / "xai_by_corruption.parquet"))

    worst = compute_worst_cases(df, threshold=0.8)
    worst.write.mode("overwrite").parquet(str(gold / "worst_cases.parquet"))

    # 3. Pandas (Arrow transfer)
    logger.info("Arrow transfer to Pandas...")
    pdf = df.toPandas()

    logger.info("Computing correlations...")
    corr_df = compute_correlations(pdf)
    corr_df.to_parquet(gold / "correlations.parquet", index=False)
    logger.info(f"Correlations:\n{corr_df.to_string(index=False)}")

    logger.info("Training meta-learner...")
    ml = train_meta_learner(pdf, gold)
    ml["results"].to_parquet(gold / "classifier_comparison.parquet", index=False)
    ml["importance"].to_parquet(gold / "feature_importance.parquet", index=False)
    logger.info(f"Meta-learner:\n{ml['results'].to_string(index=False)}")

    # 4. Summary tables
    logger.info("Building summary tables...")
    deg_pdf = deg_df.toPandas()
    adapter_pdf = adapter_df.toPandas()
    summaries = build_summary_tables(corr_df, deg_pdf, ml, adapter_pdf)

    for name, tbl in summaries.items():
        tbl.to_parquet(gold / f"{name}.parquet", index=False)
        logger.info(f"{name}:\n{tbl.to_string(index=False)}")

    # 5. Save joined
    df.write.mode("overwrite").parquet(str(gold / "joined_dataset.parquet"))

    df.unpersist()
    logger.info(f"Gold Layer completed: {gold}")
