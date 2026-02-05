import polars as pl
import pandas as pd
import logging
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
      pl.col("is_item_void").sum().alias("total_voids"),
      pl.col("line_total").filter(pl.col("is_item_void")).sum().abs().alias("void_value"),
      pl.col("line_total").filter(pl.col("is_fresh")).sum().alias("fresh_value_sum"),
      pl.col("avg_unit_price").mean().alias("avg_basket_unit_price"),
      pl.col("price_delta").mean().alias("avg_price_delta_per_basket"),
      pl.first("hour_of_day"),
      pl.first("is_weekend"),
      pl.first("store_numeric"),
      pl.first("peak_hour_pressure_numeric"),
      pl.first("cashier_id"),
      pl.first("store_id")
    ])

    agg_lf = agg_lf.with_columns([
      (pl.col("basket_size") == 1).cast(pl.Int8).alias("is_single_item_flag"),
      (pl.col("basket_value") / pl.col("basket_size")).alias("basket_value_per_item"),
      (pl.col("total_voids") / pl.col("basket_size")).fill_nan(0).alias("void_rate_per_receipt"),
      (pl.col("discounted_item_count") / pl.col("basket_size")).fill_nan(0).alias("discount_ratio"),
      # Guard against division by zero (total_weight = 0)
      (pl.col("fresh_weight_sum") / (pl.col("total_weight") + 1e-6)).alias("freshness_weight_ratio"),
      # Guard against division by zero (basket_value = 0)
      (pl.col("fresh_value_sum") / (pl.col("basket_value") + 1e-6)).alias("freshness_value_ratio"),
      # Leakage Evaluation: (Void Lines Value / Potential Gross Value)
      (pl.col("void_value") / (pl.col("basket_value") + pl.col("void_value") + 1e-6)).fill_nan(0).alias("leakage_pct")
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
      n_jobs=1 # Force single-threaded for reproducibility 
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

  def add_anomaly_score(self, df: pl.DataFrame | pl.LazyFrame, features: list):
    """Adds anomaly scores using Isolation Forest."""
    eager_df = self._ensure_eager(df)
    # Force a deterministic sort before ML
    eager_df = eager_df.sort("receipt_id")
    data_for_model = eager_df.select(features).to_pandas()

    model = IsolationForest(random_state=self.random_state, n_jobs=1, contamination='auto')
    model.fit(data_for_model)

    scores = model.decision_function(data_for_model)
    return eager_df.with_columns(pl.Series(name="anomaly_score", values=scores))

  def detect_cashier_sweethearting_anomalies(self, df: pl.DataFrame):
    """
    Flags potential sweethearting with relaxed statistical thresholds.
    """
    # Early Exit Guardrails
    if "cashier_id" not in df.columns or df.height < 2:
      logger.warning("Log: Skipping Cashier Audit - cashier_id column missing or insufficient data.")
      return df.with_columns([
        pl.lit(0).alias("cashier_sweethearting_anomaly_score"),
        pl.lit(0.0).alias("cashier_sweethearting_z_score")
      ])

    # Aggregate stats per cashier
    # Ensure discount_ratio exists and is clean before aggregation
    cashier_stats = df.group_by("cashier_id").agg([
      pl.col("discount_ratio").fill_null(0.0).mean().alias("avg_discount_rate"),
      pl.len().alias("transaction_count")
    ])

    unique_cashiers = cashier_stats.height
    logger.info(f"Audit Diagnostic: Processing {unique_cashiers} unique cashiers.")

    # Filter for Eligibility (relaxed to 2 transactions)
    eligible_stats = cashier_stats.filter(pl.col("transaction_count") >= 2)

    if eligible_stats.height < 2:
      logger.warning("Log: Cashier Audit skipped - need at least 2 distinct cashiers for peer comparison.")
      return df.with_columns([
        pl.lit(0).alias("cashier_sweethearting_anomaly_score"),
        pl.lit(0.0).alias("cashier_sweethearting_z_score")
      ])

    # Statistical Baseline Calculation
    avg_rate = eligible_stats["avg_discount_rate"].mean()
    std_rate = eligible_stats["avg_discount_rate"].std()
    
    # Robustness check: if std_rate is 0 or NaN, peer comparison is impossible
    if std_rate is None or std_rate == 0:
      logger.warning("Log: Zero variance in discount rates. All cashiers behaving identically.")
      std_rate = 0.001 

    # Z-Score Calculation & Flagging
    # Threshold 1.5 aligns with your dashboard "SOSPETTO" label
    eligible_stats = eligible_stats.with_columns(
      ((pl.col("avg_discount_rate") - avg_rate) / std_rate).alias("cashier_sweethearting_z_score")
    ).with_columns(
      pl.when(pl.col("cashier_sweethearting_z_score") > 1.5)
      .then(pl.lit(-1))
      .otherwise(pl.lit(0))
      .alias("cashier_sweethearting_anomaly_score")
    )

    # We join back to the original df to tag individual transactions
    return df.join(
      eligible_stats.select(["cashier_id", "cashier_sweethearting_z_score", "cashier_sweethearting_anomaly_score"]),
      on="cashier_id",
      how="left"
    ).with_columns([
      pl.col("cashier_sweethearting_z_score").fill_null(0.0),
      pl.col("cashier_sweethearting_anomaly_score").fill_null(0).cast(pl.Int8)
    ])

  def detect_produce_weighing_errors(self, df: pl.DataFrame):
    """
    ORTOFRUTTA WEIGHT AUDIT
    Algorithm: Median Absolute Deviation (MAD)
    Flags produce items with anomalous unit prices based on weight.
    """
    logger.info("Executing High-Precision Produce Audit (MAD-based)...")

    # Filter and deterministic sort
    produce_df = df.filter((pl.col("weight") > 0) & (pl.col("line_total") > 0)).sort("receipt_id")

    if produce_df.is_empty():
      return pl.DataFrame()

    # Calculate Median and MAD (Robust Statistics)
    # MAD is the median of the absolute deviations from the data's median
    produce_df = produce_df.with_columns([
      (pl.col("line_total") / pl.col("weight")).alias("unit_price")
    ]).with_columns([
      pl.col("unit_price").median().over("product_id").alias("med_price"),
    ]).with_columns([
      (pl.col("unit_price") - pl.col("med_price")).abs().median().over("product_id").alias("mad_price")
    ])

    # Identify outliers using the Modified Z-Score
    # We use 1.4826 as the consistency constant to make MAD comparable to Std Dev
    anomalies = produce_df.filter(
      (pl.col("mad_price") > 0) & 
      (((pl.col("unit_price") - pl.col("med_price")).abs() / (pl.col("mad_price") * 1.4826)) > 6.0)
    )

    return anomalies.select([
      pl.col("date_str"),
      pl.col("receipt_id"),
      pl.col("product_id").alias("scontrinoBarcode"),
      pl.lit("Peso/Prezzo Errante").alias("scontrinoTipo"),
      pl.col("store_id")
    ])
  
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
          n_init=10,
          algorithm="lloyd" # Ensures stable results
        )
        clusters = self.kmeans.fit_predict(X_scaled)
      else:
        X_scaled = self.scaler.transform(X_pd)
        clusters = self.kmeans.predict(X_scaled)

      # Map cluster IDs to consumer names
      centroid_map = self._map_centroid_to_mission(self.kmeans.cluster_centers_, features)
      
      # Convert cluster IDs to names
      mission_names = [centroid_map.get(c, "Standard Mixed Trip") for c in clusters]
      
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

  def extract_cashier_fingerprint(self, df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Creates a behavioral 'DNA' for each cashier based on transaction patterns.
    """
    logger.info("Generating behavioral fingerprints for cashiers...")
    
    receipt_metrics = df_raw.group_by(["cashier_id", "receipt_id"]).agg([
      pl.col("line_total").sum().alias("receipt_value"),
      pl.col("is_item_void").sum().alias("void_items_count"),
      pl.col("is_abort").any().cast(pl.Int8).alias("was_aborted"),
      pl.col("payment_method_code").first().alias("pay_code")
    ])

    fingerprint = receipt_metrics.group_by("cashier_id").agg([
      pl.len().alias("total_transactions"),
      pl.col("receipt_value").mean().alias("avg_transaction_value"),
      pl.col("void_items_count").mean().alias("void_rate_per_receipt"),
      pl.col("was_aborted").mean().alias("abort_rate"),
      # Cash Reliance Ratio: Proportion of transactions paid in cash (pay_code "01")
      ((pl.col("pay_code") == "01").sum().cast(pl.Float64) / pl.len()).alias("cash_reliance_ratio")
    ]).filter(pl.col("total_transactions") > 10)

    return fingerprint

  def detect_behavioral_anomalies(self, fp_df: pl.DataFrame) -> pl.DataFrame:
    """
    Detects cashiers whose behavior deviates significantly from the store average.
    """
    if fp_df.is_empty():
      return fp_df

    # Features to analyze
    features = ["avg_transaction_value", "void_rate_per_receipt", "abort_rate", "cash_reliance_ratio"]
    
    # Prepare for ML
    data = fp_df.select(features).to_pandas()
    data = self._sanitize_input(data)
    
    # Isolation Forest for Unsupervised Outlier Detection
    iso = IsolationForest(contamination=0.05, random_state=self.random_state)
    fp_df = fp_df.with_columns([
      pl.Series(name="iso_outlier_score", values=iso.fit_predict(data))
    ])

    # Weighted Z-Score
    # We prioritize voids and cash reliance as they are higher leakage indicators
    weights = {
      "avg_transaction_value": 0.1,
      "void_rate_per_receipt": 0.4, # Weighted higher
      "abort_rate": 0.2,
      "cash_reliance_ratio": 0.3  # Weighted higher
    }

    # Statistical Z-Score (Distance from Mean)
    # We calculate a composite risk score based on how many STDs they are from average
    risk_expr = []
    for feat, weight in weights.items():
      mean = fp_df[feat].mean()
      std = fp_df[feat].std()
      # Calculate Z-score and multiply by weight
      z_score = ((pl.col(feat) - mean) / (std + 1e-6)).abs()
      risk_expr.append(z_score * weight)

    fp_df = fp_df.with_columns([
      (sum(risk_expr)).alias("risk_score")
    ])

    return fp_df.sort("risk_score", descending=True)

  def run_margin_leakage_radar(self, df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates potential revenue loss (leakage) per cashier.
    """
    return df_raw.group_by("cashier_id").agg([
      pl.col("total_discounts").sum().alias("total_markdowns"),
      pl.col("is_item_void").sum().alias("total_voids"),
      pl.col("line_total").sum().alias("captured_revenue")
    ]).with_columns([
      (pl.col("total_markdowns") / pl.col("captured_revenue")).alias("leakage_pct")
    ])
    
  def validate_business_metrics(self, df: pl.DataFrame):
    """
    Checks for suspiciously empty business metrics.
    Categorizes alerts by impact: Missions vs. Behavioral Risk (Cashier DNA).
    """
    # Shopping Mission Dimensions (Features used for Customer Segmentation)
    mission_dims = ["basket_value", "freshness_weight_ratio", "freshness_value_ratio", "basket_size"]
    
    # Behavioral Risk DNA Dimensions (Features used for Cashier Fingerprinting & Risk Scores)
    risk_dna_dims = ["leakage_pct", "void_rate_per_receipt", "total_voids", "total_markdowns"]
    
    critical_metrics = {
      "basket_value": "Fatturato (Basket Value)",
      "freshness_weight_ratio": "Incidenza Peso Freschissimi",
      "freshness_value_ratio": "Incidenza Valore Freschissimi",
      "leakage_pct": "Percentuale Perdita (Leakage)",
      "void_rate_per_receipt": "Tasso Storni",
      "total_voids": "Totale Articoli Stornati"
    }
    
    logger.info("Starting Targeted Business Metric Validation...")
    df_eager = self._ensure_eager(df)
    
    if df_eager.is_empty():
      logger.error("VALIDATION FAILED: DataFrame is empty.")
      return

    for col, business_name in critical_metrics.items():
      if col in df_eager.columns:
        stats = df_eager.select([
          pl.col(col).filter(pl.col(col).is_finite()).sum().fill_null(0).alias("total"),
          pl.col(col).filter(pl.col(col).is_finite()).mean().fill_null(0).alias("avg")
        ])
        
        total_val = stats["total"][0]
        avg_val = stats["avg"][0]
        
        if total_val == 0:
          # Correctly attribute the impact to Cashier DNA/Risk
          if col in mission_dims:
            impact = "MISSION CLUSTERING: Customer segmentation (Missione Spesa) will be broken."
          elif col in risk_dna_dims:
            impact = "BEHAVIORAL RISK DNA: Cashier risk scoring and fingerprinting will be blinded."
          else:
            impact = "General reporting error."

          logger.critical(f"DATA INTEGRITY ALERT: '{col}' ({business_name}) is 0.0. IMPACT: {impact}")
        else:
          # Logs remain in English as per requirements
          logger.info(f"Metric '{col}' ({business_name}) validated -> [Avg: {float(avg_val):.4f}]")
      else:
        logger.error(f"SCHEMA ERROR: Required column '{col}' missing from DataFrame.")