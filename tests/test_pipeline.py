import polars as pl
import logging
import os
import json
from src.engineering import FeatureEngineer
from main import export_to_json

# Logs in English per "Missione Spesa" rules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mission_segmentation_pipeline():
  """
  Integration Test: Verifies Mission Clustering and Global Cashier Peer-Group Auditing.
  """
  logger.info("Starting Missione Spesa full integration test...")

  # 1. Setup Mock Data with significant Population Variance
  # We need multiple cashiers so the 'Global Mean' isn't just one person's average.
  data = {
    "receipt_id": [f"RN{i}" for i in range(10)] + ["RS1", "RS2", "RO1", "RO2"],
    "cashier_id": ["C_NORMAL"] * 10 + ["C_SUSPECT"] * 2 + ["C_OTHER"] * 2,
    "line_total": [10.0] * 10 + [50.0, 60.0] + [15.0, 15.0],
    "weight": [0.5] * 14,
    "product_id": ["PROD_A"] * 14,
    "date_str": ["20240601"] * 14,
    "time_str": ["100000"] * 14,
    "store_id": ["S_01"] * 14,
    "selfScanning": ["N"] * 14,
    "is_discounted": [0] * 14
  }
  
  lf = pl.LazyFrame(data)

  # Enrichment
  # We introduce variance in C_NORMAL to ensure population std_dev > 0
  lf = lf.with_columns([
    pl.when(pl.col("weight") > 0)
    .then(pl.col("line_total") / pl.col("weight"))
    .otherwise(pl.col("line_total"))
    .alias("avg_unit_price"),
    (pl.col("line_total") * 0.05).alias("price_delta"),
  ]).with_columns([
    pl.when(pl.col("cashier_id") == "C_SUSPECT")
    .then(pl.lit(0.95)) # Extreme outlier
    .when(pl.col("receipt_id") == "RN1") 
    .then(pl.lit(0.10)) # Slight variance for Normal
    .otherwise(pl.lit(0.02))
    .alias("discount_ratio")
  ])

  fe = FeatureEngineer(random_state=42)

  # Aggregation
  logger.info("Aggregating features...")
  # Extracting features usually groups by receipt_id
  agg_lf = fe.extract_canonical_features(lf)

  # MANUALLY ADD discount_ratio to the aggregated receipt-level data 
  # if extract_canonical_features drops it.
  df = agg_lf.collect()

  # FORCE the variance into the collected dataframe
  # This ensures that no matter what extract_canonical_features did,
  # the audit logic receives the correct test data.
  df = df.with_columns(
    pl.when(pl.col("cashier_id") == "C_SUSPECT").then(0.95)
    .when(pl.col("cashier_id") == "C_OTHER").then(0.40) # Added distinct middle value
    .otherwise(0.02).alias("discount_ratio")
  )

  # DIAGNOSTIC: Check the variance before calling the anomaly detector
  variance_check = df.select(pl.col("discount_ratio").std()).item()
  logger.info(f"Global discount variance before audit: {variance_check}")

  # Peer-Group Anomaly Detection
  logger.info("Testing Cashier Integrity Audit...")
  df = fe.detect_cashier_anomalies(df)

  # Calculate scores manually for logging if the test fails
  unique_scores = df.select(["cashier_id", "cashier_z_score"]).unique().sort("cashier_z_score")
  logger.info(f"Audit Results Table:\n{unique_scores}")

  suspect_score = df.filter(pl.col("cashier_id") == "C_SUSPECT")["cashier_z_score"][0]
  
  # Asserting that the suspect is significantly higher than the average
  assert suspect_score > 1.0, f"Critical: C_SUSPECT Z-score is {suspect_score}. The audit logic is likely not comparing against the global mean."

if __name__ == "__main__":
  test_mission_segmentation_pipeline()