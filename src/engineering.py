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
    except:
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
    """
    Applies a Weighted Scaling approach to ensure physical basket size 
    is as influential as monetary value.
    """
    # Clean Data
    X_df = df.select(features).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    if len(X_df) < n_clusters:
      return df.with_columns([
        pl.lit(0).alias("cluster_id"),
        pl.lit("Standard Trip").alias("shopping_mission")
      ])

    # Custom Weighted Scaling
    # We use StandardScaler first to bring everything to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    # Apply Feature Importance Weights
    # We manually boost 'basket_size' so it can compete with high-value outliers
    feature_weights = {
      "basket_size": 1.5,           # Increase sensitivity to item count
      "basket_value_per_item": 1.0, 
      "freshness_weight_ratio": 1.2 # Boost fresh-focus
    }
    
    weights_array = np.array([feature_weights.get(f, 1.0) for f in features])
    X_weighted = X_scaled * weights_array

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=15)
    clusters = kmeans.fit_predict(X_weighted)
    
    # Centroids for mapping (must inverse the scaling to interpret them)
    # Note: We un-weight then un-scale to get back to real-world units
    unweighted_centroids = kmeans.cluster_centers_ / weights_array
    actual_centroids = scaler.inverse_transform(unweighted_centroids)
    
    mapping = self._map_centroid_to_mission(actual_centroids, features)
    
    return df.with_columns([
      pl.Series(name="cluster_id", values=clusters),
      pl.Series(name="shopping_mission", values=[mapping.get(c, "Standard Trip") for c in clusters])
    ])

  def _map_centroid_to_mission(self, centroids, features):
    mission_labels = {}
    feat_map = {f: i for i, f in enumerate(features)}

    for i, center in enumerate(centroids):
      size = center[feat_map.get("basket_size", 0)]
      value_per_item = center[feat_map.get("basket_value_per_item", 0)]
      fresh = center[feat_map.get("freshness_weight_ratio", 0)]

      # B2B / Outliers (Keep high for June-style peaks)
      if size > 30 or (size > 10 and value_per_item > 150):
        mission_labels[i] = "B2B / Bulk Outlier"
      
      # Stock-up (Lowered for July/August sensitivity)
      elif size > 5:
        mission_labels[i] = "Weekly Stock-up"
      
      # Fresh Focus
      elif fresh > 0.30 and size > 1.2:
        mission_labels[i] = "Daily Fresh Pick"
      
      # Single-Item Differentiators
      elif size <= 2.5:
        if value_per_item > 45:
          mission_labels[i] = "Premium/Specialty Single-Item"
        elif value_per_item < 12:
          mission_labels[i] = "Quick Convenience"
        else:
          mission_labels[i] = "Standard Mixed Trip"
      else:
        mission_labels[i] = "Standard Mixed Trip"
            
    return mission_labels

  def determine_elbow_method(self, df, features, max_k=10):
    """Determines optimal K with a higher floor and diagnostic logging."""
    data_clean = df.select(features).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = StandardScaler().fit_transform(data_clean.values)
    
    distortions = []
    K = list(range(1, max_k + 1))
    for k in K:
      kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
      distortions.append(kmeanModel.inertia_)

    try:
      p1 = np.array([K[0], distortions[0]])
      p2 = np.array([K[-1], distortions[-1]])
      distances = [np.abs(np.cross(p2-p1, p1-np.array([K[i], distortions[i]]))) / np.linalg.norm(p2-p1) for i in range(len(K))]
      optimal_k = K[np.argmax(distances)]
      
      # For retail missions, we usually expect at least 4-5 distinct behaviors
      if optimal_k < 5:
        logger.info(f"K={optimal_k} is too low for granular missions. Nudging to K=5.")
        optimal_k = 5
                
    except Exception as e:
      logger.warning(f"⚠️ Elbow detection failed: {e}. Fallback to K=5")
      optimal_k = 5
            
    return int(optimal_k)

  def plot_mission_impact(self, insights_df, path="plots/mission_impact.png"):
    """Creates a strategic comparison between Revenue Share and Traffic Share with Italian labels."""
    os.makedirs("plots", mode=0o777, exist_ok=True)
    
    # Convert Polars to Pandas for easier plotting
    pdf = insights_df.to_pandas()
    
    # Italian translation for the legend/labels
    label_revenue = 'Quota Fatturato %'
    label_traffic = 'Quota Traffico %'
    title_text = 'Impatto Strategico Missioni: Fatturato vs Traffico'
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Set positions for the bars
    x = np.arange(len(pdf['shopping_mission']))
    width = 0.35
    
    # Plot bars with refined retail colors
    rects1 = ax1.bar(x - width/2, pdf['revenue_share_pct'], width, 
                     label=label_revenue, color='#1B5E20', alpha=0.85) # Dark Green
    rects2 = ax1.bar(x + width/2, pdf['traffic_share_pct'], width, 
                     label=label_traffic, color='#0D47A1', alpha=0.85) # Dark Blue
    
    # Styling
    ax1.set_ylabel('Percentuale (%)', fontweight='bold', fontsize=12)
    ax1.set_title(title_text, fontsize=16, pad=20, fontweight='bold')
    ax1.set_xticks(x)
    
    # Map mission names if needed, or keep as is if clusters are already descriptive
    ax1.set_xticklabels(pdf['shopping_mission'], fontweight='bold', rotation=15)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    def autolabel(rects):
      for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(path, dpi=200) # Increased DPI for professional look
    plt.close()
    logger.info(f"📈 Impact plot saved to: {path}")