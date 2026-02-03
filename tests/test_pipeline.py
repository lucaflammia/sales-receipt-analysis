import polars as pl
import logging
import os
import json
from src.engineering import FeatureEngineer
from main import export_to_json

# Ensure logs are in English as per instructions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mission_segmentation_pipeline():
  """
  Integration Test: Verifies the flow from raw mock data 
  through stable K-Means labeling and Italian report translation.
  """
  logger.info("Starting Missione Spesa integration test...")

  # Setup Mock Italian Data (Area 382)
  # We need enough variety for the clustering to have distinct centroids
  lf = pl.LazyFrame({
    "receipt_id": ["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4", "R5", "R5"] + ["R6"] * 12,
    "product_id": ["A", "B", "A", "C", "B", "D", "A", "E", "F", "G"] + ["B2B_PROD"] * 12,
    "line_total": [
      10.0, 5.0,   # R1: Standard
      20.0, 30.0,  # R2: Premium
      2.0, 3.0,    # R3: Quick Convenience
      15.0, 15.0,  # R4: Mixed
      40.0, 10.0,  # R5: Mixed
    ] + [200.0] * 12, # R6: B2B Outlier
    "weight": [0.5, 0.0, 1.0, 0.0, 0.0, 0.2, 0.4, 0.0, 1.5, 0.0] + [1.0] * 12,
    "date_str": ["20240601"] * 22,
    "time_str": ["100000"] * 22,
    "store_id": ["S_01"] * 22,
    "selfScanning": ["N"] * 22
  })

  # Mimic main.py enrichment logic
  lf = lf.with_columns(
      [pl.col("line_total").median().over("product_id").alias("Standard_Price")]
  ).with_columns([
      (pl.col("line_total") < pl.col("Standard_Price")).fill_null(False).alias("is_discounted"),
      (pl.col("line_total") - pl.col("Standard_Price")).fill_null(0.0).alias("price_delta"),
      pl.when(pl.col("weight") > 0).then(pl.col("line_total") / pl.col("weight"))
        .otherwise(pl.col("line_total")).fill_null(0.0).alias("avg_unit_price"),
  ])

  fe = FeatureEngineer(random_state=42)

  # Test Feature Extraction
  logger.info("Extracting canonical features...")
  agg_lf = fe.extract_canonical_features(lf)
  df = agg_lf.collect()

  # Clean data for ML stability
  mission_features = ["basket_size", "basket_value_per_item", "freshness_weight_ratio"]
  df = df.with_columns([
    pl.col(mission_features).fill_null(0.0),
    pl.col("basket_value").fill_null(0.0)
  ])

  # Test Elbow Integration
  logger.info("Testing Elbow Method logic...")
  optimal_k = fe.determine_elbow_method(df, features=mission_features)
  assert isinstance(optimal_k, int), "Elbow method should return an integer K"
  logger.info(f"Elbow Analysis suggested K={optimal_k}")

  # Test Stable Mission Clustering
  logger.info("Testing K-Means with centroid mapping...")
  df = fe.segment_shopping_missions(df, features=mission_features, fit_global=True)
  
  assert "shopping_mission" in df.columns, "Column 'shopping_mission' missing after clustering"
  
  # Check B2B classification (Deterministic rule)
  b2b_check = df.filter(pl.col("receipt_id") == "R6")["shopping_mission"][0]
  assert b2b_check == "B2B / Bulk Outlier", f"Expected B2B / Bulk Outlier, got {b2b_check}"

  # 6. Test Strategic Insights Generation
  grand_total_revenue = df["basket_value"].sum()
  total_receipts = df.height

  insights = (
    df.group_by("shopping_mission")
    .agg([
      pl.len().alias("receipt_count"),
      pl.col("basket_value").sum().alias("total_mission_revenue"),
      pl.col("basket_value").mean().round(2).alias("avg_trip_value")
    ])
    .with_columns([
      ((pl.col("total_mission_revenue") / grand_total_revenue) * 100).alias("revenue_share_pct"),
      ((pl.col("receipt_count") / total_receipts) * 100).alias("traffic_share_pct")
    ])
  )

  # 7. Test Italian Translation & JSON Export (Donut Chart Source)
  logger.info("Testing JSON export and Italian name translation...")
  mock_global_insights = [{"period": "2024-06", "data": insights}]
  test_json_path = "reports/test_mission_export.json"
  os.makedirs("reports", exist_ok=True)

  try:
    export_to_json(mock_global_insights, test_json_path)
    assert os.path.exists(test_json_path), "JSON export file was not created"
    
    with open(test_json_path, "r", encoding="utf-8") as f:
      exported_data = json.load(f)
      
      # Verify translation in the JSON result
      results = exported_data["monthly_insights"][0]["results"]
      mission_names = [r["shopping_mission"] for r in results]
      
      # Check if Italian values from MISSION_MAP are present
      assert any("B2B / Ingrosso Outlier" in name for name in mission_names), \
        "Italian translation failed for B2B mission"
      
      logger.info(f"Verified translated missions: {mission_names}")

  finally:
    if os.path.exists(test_json_path):
      os.remove(test_json_path)

  logger.info("✅ SUCCESS: Missione Spesa Pipeline Test Passed.")

if __name__ == "__main__":
    test_mission_segmentation_pipeline()