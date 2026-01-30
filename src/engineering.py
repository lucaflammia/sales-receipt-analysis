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
    self.scaler = StandardScaler()
    self.kmeans = None
    self.mission_labels = None

  def _ensure_eager(self, df):
    """Helper to ensure we are working with a Polars DataFrame (Eager)."""
    if isinstance(df, pl.LazyFrame):
      return df.collect()
    return df

  def _sanitize_input(self, df_pd):
    """Prevents sklearn crashes by removing Infs and extreme outliers."""
    # Replace Inf with NaN, then fill with 0
    clean_df = df_pd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Clip extreme values to float64 limits
    return clean_df.clip(lower=-1e9, upper=1e9)

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
      pl.len().alias("basket_size"),
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
      (pl.col("basket_size") == 1).cast(pl.Int8).alias("is_single_item_flag"),
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
    # Ensure we have data and fill nulls specifically for the model
    eager_df = self._ensure_eager(df)
    
    # Select features and target, filling nulls with 0 to prevent dropping all rows
    data = eager_df.select(features + [target]).fill_null(0.0).fill_nan(0.0)
    
    if data.height == 0:
      logger.warning("Empty data for RF feature selection. Returning empty dict.")
      return {}

    X = data.select(features).to_pandas()
    y = data.select(target).to_pandas().values.ravel()

    # Clean any stray Infs
    X = X.replace([np.inf, -np.inf], 0.0)

    rf = RandomForestRegressor(
      n_estimators=100, # Reduced for speed, keep 500 if latency isn't an issue
      random_state=self.random_state,
      n_jobs=-1 
    )
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    logger.info(f"Stable RF Importances:\n{importances.head(top_n)}")
    
    return importances.head(top_n).to_dict()

  def select_features_lasso(self, df, features, target):
    eager_df = self._ensure_eager(df)
    data = eager_df.select(features + [target]).fill_null(0.0).fill_nan(0.0)
    
    if data.height < 5: # Lasso needs a few samples for CV
      return {}

    X = data.select(features).to_pandas().replace([np.inf, -np.inf], 0.0)
    y = data.select(target).to_pandas().values.ravel()

    try:
      X_scaled = StandardScaler().fit_transform(X)
      lasso = LassoCV(cv=5, random_state=self.random_state).fit(X_scaled, y)
      coefs = pd.Series(lasso.coef_, index=features)
      return coefs[coefs.abs() > 1e-3].to_dict()
    except Exception as e:
      logger.warning(f"Lasso feature selection failed: {e}")
      return {}

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
    """Calculates VIF with a floor on sample size to prevent empty frames."""
    eager_df = self._ensure_eager(df)
    
    # Ensure floor of 100 rows or total rows if less than 100
    sample_n = max(min(len(eager_df), 100), int(len(eager_df) * sample_fraction))
    df_sampled = eager_df.sample(n=sample_n, seed=self.random_state)

    df_pd = df_sampled.select(features).to_pandas()
    # VIF Guardrail: Remove NaNs and Infs
    # Ratios (like fresh_weight / total_weight) often produce NaNs or Inf
    df_pd = df_pd.replace([np.inf, -np.inf], np.nan).dropna()

    if df_pd.empty or len(df_pd) < len(features) + 1:
      logging.warning("Insufficient clean data for VIF. Check for constants or high null counts.")
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

  def segment_shopping_missions(self, df, features, n_clusters=6, fit_global=False):
    """
    Segments receipts into missions using a hybrid approach:
    1. Deterministic Rule for B2B/Bulk Outliers.
    2. K-Means Clustering for the remaining consumer missions.
    """
    # Define the B2B Rule
    b2b_condition = (
      (pl.col("basket_size") > 30) | 
      ((pl.col("basket_size") > 10) & (pl.col("basket_value_per_item") > 150))
    )

    # Flag B2B Outliers
    df = df.with_columns(
      pl.when(b2b_condition)
      .then(pl.lit("B2B / Bulk Outlier"))
      .otherwise(pl.lit("PENDING"))
      .alias("shopping_mission")
    )

    # Extract data that needs clustering
    pending_mask = df["shopping_mission"] == "PENDING"
    df_pending = df.filter(pending_mask)

    if df_pending.height > 0:
      # Prepare features for the pending subset
      X_pd = self._sanitize_input(df_pending.select(features).to_pandas())
      
      # Determine a safe number of clusters for the available data
      # We target (n_clusters - 1) because B2B is already its own category
      k_target = min(n_clusters - 1, df_pending.height)
      k_target = max(2, k_target)

      if fit_global or self.kmeans is None:
        logger.info(f"Fitting consumer centroids for k={k_target}...")
        X_scaled = self.scaler.fit_transform(X_pd)
        self.kmeans = KMeans(
          n_clusters=k_target, 
          init='k-means++', 
          random_state=self.random_state, 
          n_init=10
        )
        clusters = self.kmeans.fit_predict(X_scaled)
      else:
        X_scaled = self.scaler.transform(X_pd)
        clusters = self.kmeans.predict(X_scaled)

      # Map cluster IDs to consumer names
      # Standard logic mapping based on project requirements
      consumer_map = {
        0: "Standard Mixed Trip",
        1: "Quick Convenience",
        2: "Weekly Stock-up",
        3: "Daily Fresh Pick",
        4: "Premium/Specialty Single-Item"
      }
      
      # Convert cluster IDs to names
      mission_names = [consumer_map.get(c, "Standard Mixed Trip") for c in clusters]
      
      # Create a temporary mapping dataframe to avoid ShapeError
      # We use the unique receipt_id from the pending subset to align the names
      mapping_df = pl.DataFrame({
        "receipt_id": df_pending["receipt_id"],
        "clustered_name": mission_names
      })

      # Join the mapping back to the main dataframe
      df = df.join(mapping_df, on="receipt_id", how="left")
      
      # Finalize the shopping_mission column
      df = df.with_columns(
        pl.when(pl.col("shopping_mission") == "PENDING")
        .then(pl.col("clustered_name"))
        .otherwise(pl.col("shopping_mission"))
        .alias("shopping_mission")
      ).drop("clustered_name")

    return df

  def _map_centroid_to_mission(self, centers, features):
    """
    Heuristic mapping of K-Means centroids to Mission names.
    This ensures Cluster 0 in January is the same 'Mission' as Cluster 0 in February.
    """
    mapping = {}
    feat_idx = {feat: i for i, feat in enumerate(features)}
    
    for i, center in enumerate(centers):
      size = center[feat_idx["basket_size"]]
      val_per_item = center[feat_idx["basket_value_per_item"]]
      fresh_ratio = center[feat_idx["freshness_weight_ratio"]]
      
      if size > 15:
        mapping[i] = "B2B / Bulk Outlier"
      elif size > 5:
        mapping[i] = "Weekly Stock-up"
      elif fresh_ratio > 0.3:
        mapping[i] = "Daily Fresh Pick"
      elif size <= 2.5 and val_per_item > 45:
        mapping[i] = "Premium/Specialty Single-Item"
      elif size <= 2.5 and val_per_item < 12:
        mapping[i] = "Quick Convenience"
      else:
        mapping[i] = "Standard Mixed Trip"
    return mapping

  def determine_elbow_method(self, df, features, max_k=10):
    """Safely calculates the optimal K using sanitized data."""
    logger.info("Running Elbow Method...")
    X_pd = self._sanitize_input(df.select(features).to_pandas())
    X_scaled = self.scaler.fit_transform(X_pd)
    
    # Calculate the mathematical limit for clusters based on sample size
    # n_samples must be > n_clusters
    n_samples = X_pd.shape[0]
    limit = min(max_k, n_samples - 1)
    
    if limit < 2:
        logger.warning(f"Insufficient samples ({n_samples}) for Elbow Method. Returning default K=6.")
        return 6 
    
    wcss = []

    for i in range(2, limit + 1):
      km = KMeans(n_clusters=i, init='k-means++', random_state=self.random_state, n_init=10)
      km.fit(X_scaled)
      wcss.append(km.inertia_)
    
    # Logic to find the 'elbow' would go here (e.g., Kneed)
    # For Missione Spesa, we return 6 as the target mission count
    return 6

  def plot_mission_impact(self, df_pd, path, title="Impatto Missioni di Spesa"):
    """
    Generates a dual-axis chart for Business Stakeholders:
    - Bars: Total Revenue (Fatturato)
    - Line: Traffic Share (Incidenza Traffico %)
    """
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Axis 1: Revenue (Bars)
    sns.barplot(
      data=df_pd, 
      x="shopping_mission", 
      y="total_mission_revenue", 
      ax=ax1, 
      hue="shopping_mission", 
      palette="viridis", 
      legend=False
    )
    ax1.set_ylabel("Fatturato Totale (€)", fontweight='bold')
    ax1.set_xlabel("Missione di Spesa", fontweight='bold')
    plt.xticks(rotation=15, ha='right')

    # Axis 2: Traffic Share (Line)
    ax2 = ax1.twinx()
    sns.lineplot(
      data=df_pd, 
      x="shopping_mission", 
      y="traffic_share_pct", 
      ax=ax2, 
      color="#C62828", 
      marker="o", 
      linewidth=2.5,
      label="Quota Traffico %"
    )
    ax2.set_ylabel("Incidenza sul Traffico (%)", fontweight='bold', color="#C62828")
    ax2.set_ylim(0, df_pd["traffic_share_pct"].max() * 1.2)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()