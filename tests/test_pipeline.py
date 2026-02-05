import polars as pl
import logging
from src.engineering import FeatureEngineer

# Logs in English per "Missione Spesa" rules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mission_segmentation_pipeline():
  """
  Integration Test: Verifies Mission Clustering and Behavioral Risk DNA (Cashier DNA) Auditing.
  """
  logger.info("Starting Missione Spesa / Cashier DNA full integration test...")

  # INCREASE POPULATION: Add 5 more normal cashiers to lower the global standard deviation
  normal_cashiers = [f"C_NORM_{i}" for i in range(5)]
  all_cashiers = ["C_NORMAL"] * 10 + ["C_SUSPECT"] * 2 + ["C_OTHER"] * 2
  for nc in normal_cashiers:
    all_cashiers += [nc] * 2

  data = {
    "receipt_id": [f"R_{i}" for i in range(len(all_cashiers))],
    "cashier_id": all_cashiers,
    "line_total": [10.0] * len(all_cashiers),
    "weight": [0.5] * len(all_cashiers),
    "product_id": ["PROD_A"] * len(all_cashiers),
    "date_str": ["20260205"] * len(all_cashiers),
    "time_str": ["100000"] * len(all_cashiers),
    "store_id": ["S_01"] * len(all_cashiers),
    "selfScanning": ["N"] * len(all_cashiers),
    "is_discounted": [0] * len(all_cashiers),
    "is_item_void": [False] * len(all_cashiers),
    "is_abort": [False] * len(all_cashiers)
  }
  
  # Create LazyFrame and add missing columns for FeatureEngineer requirements
  lf = pl.LazyFrame(data).with_columns([
    (pl.col("line_total") * 0.05).alias("price_delta"),
    pl.lit(5.0).alias("avg_unit_price"),
    pl.col("line_total").alias("total_discounts") # Required for Leakage Radar
  ])

  fe = FeatureEngineer(random_state=42)

  # Feature Extraction & Aggregation
  logger.info("Aggregating features and calculating ratios...")
  agg_lf = fe.extract_canonical_features(lf)
  df = agg_lf.collect()

  # Business Metric Validation (Testing the Inf/Zero fix)
  # This will trigger the logger.info for validated metrics
  fe.validate_business_metrics(df)

  # Cashier Behavioral Risk DNA Logic
  logger.info("Testing Behavioral Risk DNA (Sweethearting) Audit...")
  
  # Force specific discount ratios to test the Z-Score engine
  df = df.with_columns(
    pl.when(pl.col("cashier_id") == "C_SUSPECT").then(0.95)
    .when(pl.col("cashier_id") == "C_OTHER").then(0.05)
    .otherwise(0.01).alias("discount_ratio")
  )

  # Run the specific sweethearting detector
  df = fe.detect_cashier_sweethearting_anomalies(df)

  # 5. DIAGNOSTIC: Risk DNA Results
  unique_scores = df.select([
    "cashier_id", 
    "cashier_sweethearting_z_score", 
    "cashier_sweethearting_anomaly_score"
  ]).unique().sort("cashier_sweethearting_z_score")
  
  logger.info(f"Cashier Risk DNA Results:\n{unique_scores}")

  # Assertions
  suspect_data = df.filter(pl.col("cashier_id") == "C_SUSPECT").head(1)
  suspect_z = suspect_data["cashier_sweethearting_z_score"][0]
  suspect_flag = suspect_data["cashier_sweethearting_anomaly_score"][0]

  # Verify suspect is caught (Z > 1.5 and flag is -1)
  assert suspect_z > 1.5, f"DNA Failure: C_SUSPECT Z-score is {suspect_z}, too low for peer outlier."
  assert suspect_flag == -1, "DNA Failure: C_SUSPECT was not flagged as an anomaly (expected -1)."
  
  logger.info("SUCCESS: Integration test passed. Risk DNA and Validation logic are stable.")

if __name__ == "__main__":
  test_mission_segmentation_pipeline()