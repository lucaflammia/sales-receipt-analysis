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
from sklearn.cluster import KMeans
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

  def extract_canonical_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
    """
      Implements the requirements:
      1. Base Features (Baseline parity)
      2. Extra Features (Weight Handling, Peak Pressure, etc.)
    """
    logger.info("Replicating Canonical Base + Adding Extra Features...")

    # Base Features (Baseline)
    lf = lf.with_columns([
      pl.col("date_str").cast(pl.String).str.to_date("%Y%m%d", strict=False),
      pl.col("time_str").cast(pl.String).str.slice(0, 2).cast(pl.Int32).alias("hour_of_day"),
      pl.col("selfScanning").replace({"Y": 1, "N": 0}).cast(pl.Int8)
    ]).with_columns([
      pl.col("date_str").dt.weekday().alias("day_of_week"),
      pl.col("date_str").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
      # Categorical hashing for ML
      pl.col("store_id").cast(pl.String).hash().mod(1000).alias("store_numeric"),
    ])

    # Weight Handling & Peak Pressure
    lf = lf.with_columns([
      pl.col("weight").fill_null(0.0).cast(pl.Float64),
      pl.when((pl.col("hour_of_day").is_between(12, 14)) | (pl.col("hour_of_day").is_between(18, 20)))
        .then(pl.lit(1)).otherwise(pl.lit(0)).alias("peak_hour_pressure_numeric")
    ])
    
    # Freshness Logic
    lf = lf.with_columns((pl.col("weight") > 0).alias("is_fresh"))

    agg_lf = lf.group_by("receipt_id").agg([
      pl.col("line_total").sum().alias("basket_value"),
      pl.col("product_id").n_unique().alias("basket_size"),
      pl.col("is_fresh").any().cast(pl.Int8).alias("has_fresh_produce"),
      pl.col("is_discounted").sum().alias("discounted_item_count"),
      pl.col("weight").filter(pl.col("is_fresh")).sum().alias("fresh_weight_sum"),
      pl.col("weight").sum().alias("total_weight"),
      pl.col("line_total").filter(pl.col("is_fresh")).sum().alias("fresh_value_sum"),
      pl.col("avg_unit_price").mean().alias("avg_basket_unit_price"),
      pl.col("price_delta").mean().alias("avg_price_delta_per_basket"),
      pl.first("hour_of_day"),
      pl.first("is_weekend"),
      pl.first("store_numeric"),
      pl.first("peak_hour_pressure_numeric")
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
      pl.col("freshness_value_ratio").cast(pl.Float64).fill_nan(0),
      pl.col("avg_price_delta_per_basket").fill_null(0) # Center the delta if missing
    ])
    return agg_lf

  def map_severity(self, df: pl.DataFrame):
    """Maps scores to critical/warning/info to align with reference report labels."""
    scores = df.select("anomaly_score").to_series()
    # Lower score = more anomalous
    q01 = scores.quantile(0.01)
    q05 = scores.quantile(0.05)

    return df.with_columns(
      pl.when(pl.col("anomaly_score") <= q01).then(pl.lit("critical"))
      .when(pl.col("anomaly_score") <= q05).then(pl.lit("warning"))
      .otherwise(pl.lit("info"))
      .alias("severity")
    )

  def select_features_rf(self, df, features, target, top_n=3):
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

  def calculate_vif(self, df: pl.DataFrame | pl.LazyFrame, features: dict, sample_fraction: float = 0.1):
    """Calculates VIF with data cleaning for statsmodels compatibility."""
    logging.info(f"Calculating VIF for features: {[k for k in features.keys()]}")

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
    Dynamically generates anomaly plots for the most relevant features.
    Prioritizes top features if importance scores are provided.
    """
    os.makedirs("plots", exist_ok=True)

    # Prepare Data
    eager_df = self._ensure_eager(df)
    if "anomaly_score" not in eager_df.columns:
      logging.warning("No 'anomaly_score' column found. Skipping plot.")
      return

    # Sample for performance, but ensure we have enough data
    sample_size = min(len(eager_df), 5000)
    pdf = eager_df.sample(n=sample_size, seed=self.random_state).to_pandas()

    # Logic to determine "Most Relevant" features
    feat_scores = None
    if isinstance(features, dict):
      # Sort features by importance score descending and take top 4
      sorted_feats = sorted(features.items(), key=lambda x: x[1], reverse=True)
      plot_cols = [f[0] for f in sorted_feats[:4]]
      feat_scores = {f[0]: f[1] for f in sorted_feats[:4]}
    elif isinstance(features, list):
      plot_cols = features[:4] # Cap at top 4 for readability
    else:
      # Fallback: find numerical columns that aren't the score or ID
      plot_cols = [c for c in pdf.select_dtypes(include=['number']).columns 
                    if c not in ["anomaly_score", "receipt_id", "store_numeric"]][:4]

    num_features = len(plot_cols)
    if num_features < 2:
      logging.warning("Not enough features selected for plotting.")
      return

    # 3D Plotting (The "Money Shot")
    if num_features >= 3:
      path_3d = path.replace(".png", "_3d.png")
      fig = plt.figure(figsize=(10, 8))
      ax = fig.add_subplot(111, projection="3d")

      # Use the top 3 features
      scatter = ax.scatter(
        pdf[plot_cols[0]],
        pdf[plot_cols[1]],
        pdf[plot_cols[2]],
        c=pdf["anomaly_score"],
        cmap="rocket",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5
      )

      # Labels with Importance %
      labels = []
      for c in plot_cols[:3]:
        label = c.replace("_", " ").title()
        if feat_scores:
          label += f" ({feat_scores[c]:.1f}%)"
        labels.append(label)

      ax.set_xlabel(labels[0])
      ax.set_ylabel(labels[1])
      ax.set_zlabel(labels[2])
      
      plt.title("3D Anomaly Distribution (Top Consensus Features)")
      fig.colorbar(scatter, ax=ax, label="Anomaly Score", pad=0.1)
      plt.savefig(path_3d, dpi=150, bbox_inches='tight')
      plt.close()
      logging.info(f"✨ Top 3 features 3D plot saved: {path_3d}")

    # Optimized 2.D Pair-wise Plotting
    # We only plot the most significant combinations (Max 3)
    combinations = list(itertools.combinations(plot_cols, 2))[:3]
    num_plots = len(combinations)

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
      axes = [axes]

    for i, (feat_x, feat_y) in enumerate(combinations):
      sns.scatterplot(
        data=pdf,
        x=feat_x,
        y=feat_y,
        hue="anomaly_score",
        palette="rocket",
        size="anomaly_score",
        sizes=(10, 100),
        alpha=0.7,
        ax=axes[i]
      )
      
      # Enhanced titling
      title = f"{feat_y.replace('_', ' ').title()}\nvs {feat_x.replace('_', ' ').title()}"
      axes[i].set_title(title, fontweight='bold')
      axes[i].set_xlabel(feat_x.replace("_", " ").title())
      axes[i].set_ylabel(feat_y.replace("_", " ").title())
      axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"📊 Top 2D pairwise plots saved: {path}")

  def segment_shopping_missions(self, df, features, n_clusters=4):
    """Applies K-Means and maps results to missions."""
    if df.height < n_clusters:
      logger.warning("Not enough data points to form requested clusters.")
      return df

    # Standardize for equal weighting
    scaler = StandardScaler()
    X = df.select(features).to_numpy()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add IDs and Map Labels
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    mapping = self._map_centroid_to_mission(centroids, features)
    
    return df.with_columns([
      pl.Series(name="cluster_id", values=clusters),
      pl.Series(name="shopping_mission", values=[mapping[c] for c in clusters])
    ])

  def _map_centroid_to_mission(self, centroids, features):
    """Translates numeric centroids into business missions with safety guards."""
    mission_labels = {}
    feat_map = {f: i for i, f in enumerate(features)}

    # Extract indices with fallback to None if feature wasn't used in clustering
    idx_size = feat_map.get("basket_size")
    idx_fresh = feat_map.get("freshness_weight_ratio")
    idx_value = feat_map.get("basket_value_per_item")

    for i, center in enumerate(centroids):
      # Get values or neutral defaults if feature is missing
      size = center[idx_size] if idx_size is not None else 5
      fresh = center[idx_fresh] if idx_fresh is not None else 0.2
      value = center[idx_value] if idx_value is not None else 10

      # Logic based on Italian retail benchmarks
      if size > 12:
        mission_labels[i] = "Weekly Stock-up"
      elif fresh > 0.45:
        mission_labels[i] = "Daily Fresh Trip"
      elif size <= 3 and value > 15:
        mission_labels[i] = "Convenience Stop"
      else:
        mission_labels[i] = "Standard Fill"
            
    return mission_labels

  def plot_elbow_method(self, df, features, max_k=10, path="plots/elbow.png"):
    """Determines the optimal K."""
    X = StandardScaler().fit_transform(df.select(features).to_numpy())
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
      kmeanModel = KMeans(n_clusters=k, n_init=10).fit(X)
      distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(10,6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal k')
    plt.savefig(path)
    plt.close()
