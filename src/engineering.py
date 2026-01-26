import polars as pl
import pandas as pd
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)


class FeatureEngineer:
  def __init__(self, random_state=42):
      self.random_state = random_state

  def _ensure_eager(self, df):
      """Helper to ensure we are working with a Polars DataFrame (Eager)."""
      if isinstance(df, pl.LazyFrame):
          return df.collect()
      return df

  def calculate_vif(
    self,
    df: pl.DataFrame | pl.LazyFrame,
    features: list,
    sample_fraction: float = 0.1,
  ):
    """Calculates VIF with data cleaning for statsmodels compatibility."""
    logging.info(f"Calculating VIF for features: {features}")

    # 1. Sample and Materialize
    df_sampled = df.sample(fraction=sample_fraction, seed=self.random_state)
    df_pd = self._ensure_eager(df_sampled).select(features).to_pandas()

    # VIF Guardrail: Remove NaNs and Infs
    # Ratios (like fresh_weight / total_weight) often produce NaNs or Inf
    df_pd = df_pd.replace([np.inf, -np.inf], np.nan).dropna()

    if df_pd.empty or len(df_pd) < len(features) + 1:
      logging.warning("Not enough clean data points for VIF calculation.")
      return {}

    # Standard VIF Calculation
    X = add_constant(df_pd)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
      variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    return (
      vif_data[vif_data["feature"] != "const"]
      .set_index("feature")["VIF"]
      .to_dict()
    )

  def calculate_pearson_matrix(self, df: pl.DataFrame | pl.LazyFrame, features: list):
    """Calculates Pearson correlation natively in Polars."""
    eager_df = self._ensure_eager(df)
    return eager_df.select(features).corr()

  def extract_shopping_mission_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    logging.info("Extracting shopping mission features...")

    # Ensure Weight is handled (Standardized Name)
    lf = lf.with_columns(
      pl.col("Weight").fill_null(0.0).cast(pl.Float64)
    )

    # Temporal Features (Standardized Names)
    lf = lf.with_columns([
      pl.col("time_str").str.to_time(format="%H:%M:%S", strict=False).alias("time_obj"),
      pl.col("date_str").str.to_date(format="%Y-%m-%d", strict=False).alias("date_obj")
    ]).with_columns([
      pl.col("time_obj").dt.hour().fill_null(0).alias("hour"),
      pl.col("date_obj").dt.weekday().fill_null(1).alias("day_of_week"),
      pl.col("date_obj").dt.month().fill_null(1).alias("month"),
      pl.col("date_obj").dt.year().fill_null(2024).alias("year"),
    ])

    # Peak Pressure
    lf = lf.with_columns(
      pl.when((pl.col("hour").is_between(8, 10)) | (pl.col("hour").is_between(12, 14)) | (pl.col("hour").is_between(17, 19)))
      .then(pl.lit("peak")).otherwise(pl.lit("off-peak")).alias("peak_hour_pressure")
    )

    # Freshness Logic
    lf = lf.with_columns((pl.col("Weight") > 0).alias("is_fresh"))

    # Aggregation
    agg_lf = lf.group_by("receipt_id").agg([
      pl.col("line_total").sum().alias("basket_value"),
      pl.col("Product SKU").n_unique().alias("basket_size"),
      pl.col("is_fresh").any().cast(pl.Int8).alias("has_fresh_produce"),
      pl.col("Weight").filter(pl.col("is_fresh")).sum().alias("fresh_weight_sum"),
      pl.col("Weight").sum().alias("total_weight"),
      pl.col("line_total").filter(pl.col("is_fresh")).sum().alias("fresh_value_sum"),
      pl.first("hour"),
      pl.first("day_of_week"),
      pl.first("month"),
      pl.first("year")
    ])

    # Update this section in extract_shopping_mission_features
    agg_lf = agg_lf.with_columns([
      (pl.col("basket_value") / pl.col("basket_size")).alias("value_per_item"),
      
      # Use .fill_nan(0) and .replace(np.inf, 0) logic
      (pl.col("fresh_weight_sum") / pl.col("total_weight"))
        .fill_nan(0)
        .alias("freshness_weight_ratio"),
          
      (pl.col("fresh_value_sum") / pl.col("basket_value"))
        .fill_nan(0)
        .alias("freshness_value_ratio")
    ]).with_columns([
        # Secondary guardrail for potential Infinity
        pl.col("freshness_weight_ratio").cast(pl.Float64).fill_nan(0),
        pl.col("freshness_value_ratio").cast(pl.Float64).fill_nan(0)
    ])

    return agg_lf

  def add_anomaly_score(self, df: pl.DataFrame | pl.LazyFrame, features: list, model_path: str = None):
    """Adds anomaly scores using Isolation Forest."""
    eager_df = self._ensure_eager(df)
    data_for_model = eager_df.select(features).to_pandas()

    model = IsolationForest(random_state=self.random_state)
    model.fit(data_for_model)

    scores = model.decision_function(data_for_model)
    return eager_df.with_columns(pl.Series(name="anomaly_score", values=scores))
