import polars as pl
import pandas as pd
import logging
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
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

  def select_features_rf(self, df, features, target, top_n=5):
    # Sort for determinism before converting to pandas
    eager_df = self._ensure_eager(df).sort("receipt_id").drop_nulls()
    
    X = eager_df.select(features).to_pandas()
    y = eager_df.select(target).to_pandas().values.ravel()

    # Replace Infinity with NaN and then drop them
    # Random Forest cannot handle 'inf'
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]

    # Increased n_estimators for better stability
    rf = RandomForestRegressor(
      n_estimators=500, 
      random_state=self.random_state,
      n_jobs=-1 # Speed up with parallel processing
    )
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    logger.info(f"Stable RF Importances:\n{importances.head(top_n)}")
    
    return importances.head(top_n).to_dict()

  def select_features_lasso(self, df, features, target):
    eager_df = self._ensure_eager(df).drop_nulls()
    X = eager_df.select(features).to_pandas()
    y = eager_df.select(target).to_pandas().values.ravel()

    # Clean Infinity
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]

    X_scaled = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, random_state=self.random_state).fit(X_scaled, y)
    coefs = pd.Series(lasso.coef_, index=features)
    return coefs[coefs.abs() > 1e-3].to_dict()

  def get_consensus_features(self, rf_results: dict, lasso_results: dict):
    """
    Returns a dictionary of features identified by both models,
    weighted by their normalized importance.
    """
    rf_set = set(rf_results.keys())
    lasso_set = set(lasso_results.keys())
    consensus_keys = list(rf_set.intersection(lasso_set))

    # Normalize scores to 0-100 for comparison
    consensus_with_scores = {}
    
    # Simple normalization helper
    rf_max = max(rf_results.values()) if rf_results else 1
    ls_max = max(abs(v) for v in lasso_results.values()) if lasso_results else 1

    for feat in consensus_keys:
      # Average of normalized RF importance and normalized Lasso magnitude
      score = (
        (rf_results[feat] / rf_max * 50) + 
        (abs(lasso_results[feat]) / ls_max * 50)
      )
      consensus_with_scores[feat] = round(score, 2)

    # Sort by the new consensus score
    sorted_consensus = dict(sorted(consensus_with_scores.items(), key=lambda x: x[1], reverse=True))
    
    logging.info(f"Consensus features with confidence scores: {sorted_consensus}")
    return sorted_consensus

  def calculate_vif(self, df: pl.DataFrame | pl.LazyFrame, features: list, sample_fraction: float = 0.1):
    """Calculates VIF with data cleaning for statsmodels compatibility."""
    logging.info(f"Calculating VIF for features: {features}")

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

    agg_lf = lf.group_by("receipt_id").agg([
      pl.col("line_total").sum().alias("basket_value"),
      pl.col("Product SKU").n_unique().alias("basket_size"),
      pl.col("is_fresh").any().cast(pl.Int8).alias("has_fresh_produce"),
      pl.col("is_discounted").sum().alias("discounted_item_count"),
      pl.col("Weight").filter(pl.col("is_fresh")).sum().alias("fresh_weight_sum"),
      pl.col("Weight").sum().alias("total_weight"),
      pl.col("line_total").filter(pl.col("is_fresh")).sum().alias("fresh_value_sum"),
      pl.col("avg_unit_price").mean().alias("avg_basket_unit_price"),
      pl.first("hour"),
      pl.first("day_of_week"),
      pl.first("month"),
      pl.first("year")
    ])

    agg_lf = agg_lf.with_columns([
      (pl.col("basket_value") / pl.col("basket_size")).alias("basket_value_per_item"),
      (pl.col("discounted_item_count") / pl.col("basket_size")).fill_nan(0).alias("discount_ratio"),
      (pl.col("fresh_weight_sum") / pl.col("total_weight")).fill_nan(0).alias("freshness_weight_ratio"),
      (pl.col("fresh_value_sum") / pl.col("basket_value")).fill_nan(0).alias("freshness_value_ratio")
    ]).with_columns([
      # Guardrails for potential Infinity or NaN from divisions
      pl.col("basket_value_per_item").fill_nan(0).fill_null(0),
      pl.col("avg_basket_unit_price").fill_nan(0).fill_null(0),
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

  def plot_anomalies(self, df, features=None, path="plots/anomalies.png"):
    """
    Dynamically generates anomaly plots.
    Handles 'features' as either a list of names or a dict of {name: score}.
    """
    os.makedirs("plots", exist_ok=True)

    eager_df = self._ensure_eager(df)
    pdf = eager_df.sample(
      n=min(len(eager_df), 2000), seed=self.random_state
    ).to_pandas()

    if isinstance(features, dict):
      plot_cols = list(features.keys())
      feat_scores = list(features.values())
    elif isinstance(features, list):
      plot_cols = features
      feat_scores = None
    else:
      plot_cols = [c for c in pdf.columns if c != "anomaly_score"]
      feat_scores = None

    num_features = len(plot_cols)

    if num_features < 2:
      logging.warning("Not enough features for dynamic plotting.")
      return

    # 3D Plotting (If exactly 3 features)
    if num_features == 3:
      path_3d = path.replace(".png", "_3d.png")
      fig = plt.figure(figsize=(10, 7))
      ax = fig.add_subplot(111, projection="3d")

      scatter = ax.scatter(
        pdf[plot_cols[0]],
        pdf[plot_cols[1]],
        pdf[plot_cols[2]],
        c=pdf["anomaly_score"],
        cmap="rocket",
        alpha=0.6,
      )

      # Apply labels with confidence scores if available
      if feat_scores:
        ax.set_xlabel(f"{plot_cols[0].title()} ({feat_scores[0]}%)")
        ax.set_ylabel(f"{plot_cols[1].title()} ({feat_scores[1]}%)")
        ax.set_zlabel(f"{plot_cols[2].title()} ({feat_scores[2]}%)")
      else:
        ax.set_xlabel(plot_cols[0].title())
        ax.set_ylabel(plot_cols[1].title())
        ax.set_zlabel(plot_cols[2].title())

      plt.title("3D Anomaly Map (Consensus Features)")
      fig.colorbar(scatter, ax=ax, label="Anomaly Score")
      plt.savefig(path_3d)
      plt.close()
      logging.info(f"3D anomaly plot saved to {path_3d}")

    # Dynamic 2D Pair-wise Plotting
    combinations = list(itertools.combinations(plot_cols, 2))
    num_plots = len(combinations)

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
      axes = [axes]

    for i, (feat_x, feat_y) in enumerate(combinations):
      sns.scatterplot(
        data=pdf,
        x=feat_x,
        y=feat_y,
        hue="anomaly_score",
        palette="rocket",
        ax=axes[i],
      )
      axes[i].set_title(f"{feat_y} vs {feat_x}")
      axes[i].set_xlabel(feat_x.replace("_", " ").title())
      axes[i].set_ylabel(feat_y.replace("_", " ").title())

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"Dynamic multi-panel anomaly plot saved to {path}")
