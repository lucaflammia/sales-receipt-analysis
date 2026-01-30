import polars as pl
import pandas as pd
import logging
import os
import pytest
import numpy as np
from src.engineering import FeatureEngineer
from main import generate_global_summary

logger = logging.getLogger(__name__)

def test_mission_segmentation_pipeline():
  """
  Integration Test: Verifies the flow from raw mock data 
  through K-Means optimization (Missione Spesa) and Anomaly Auditing.
  """
  # Setup Mock Italian Data
  # R6 triggers the B2B rule: (Items > 10 and Value/Item > 150)
  lf = pl.LazyFrame({
    "receipt_id": ["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4", "R5", "R5"] + ["R6"] * 12,
    "product_id": ["A", "B", "A", "C", "B", "D", "A", "E", "F", "G"] + ["B2B_PROD"] * 12,
    "line_total": [
        10.0, 5.0,   # R1: Standard
        20.0, 30.0,  # R2: Premium
        5.0, 5.0,    # R3: Convenience
        15.0, 15.0,  # R4: Mixed
        40.0, 10.0,  # R5: Mixed
    ] + [200.0] * 12, # R6: B2B Outlier (12 items * 200 = 2400 total)
    "weight": [0.5, 0.0, 1.0, 0.0, 0.0, 0.2, 0.4, 0.0, 1.5, 0.0] + [1.0] * 12,
    "date_str": ["20240601"] * 22,
    "time_str": ["100000"] * 22,
    "store_id": ["S_01"] * 22,
    "selfScanning": ["N"] * 22
  })

  # Item-Level Enrichment
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
  agg_lf = fe.extract_canonical_features(lf)
  df = agg_lf.collect()

  # Fill nulls for ML stability
  mission_features = ["basket_size", "basket_value_per_item", "freshness_weight_ratio"]
  df = df.with_columns([
    pl.col(mission_features).fill_null(0.0),
    pl.col("basket_value").fill_null(0.0)
  ])

  # --- TEST ELBOW INTEGRATION ---
  # We cap the search range to (number of samples - 1) to avoid sklearn ValueError
  # In this mock case, height is 6, so we test K up to 5.
  max_k_test = min(10, df.height - 1) 
  
  if max_k_test > 1:
    # Pass a range or a max_k if your determine_elbow_method supports it, 
    # otherwise ensure determine_elbow_method handles small data internally.
    optimal_k = fe.determine_elbow_method(df, features=mission_features)
    assert isinstance(optimal_k, int)
    logger.info(f"Test Elbow Result: {optimal_k}")
  else:
    logger.warning("Skipping Elbow test: insufficient mock data.")

  # Test Mission Clustering (fit_global=True to mimic main.py start)
  # Ensure n_clusters here is also safe for the mock data size
  df = fe.segment_shopping_missions(df, features=mission_features, fit_global=True)
  
  assert "shopping_mission" in df.columns
  
  # Check R6 classification (Should be deterministic B2B)
  b2b_check = df.filter(pl.col("receipt_id") == "R6")["shopping_mission"][0]
  assert b2b_check == "B2B / Bulk Outlier", f"Expected B2B, but got {b2b_check}"

  # Test Consensus & Anomaly Score
  rf_imp = fe.select_features_rf(df, features=mission_features, target="basket_value")
  ls_imp = fe.select_features_lasso(df, features=mission_features, target="basket_value")
  consensus = fe.get_consensus_features(rf_imp, ls_imp)

  if consensus:
    df = fe.add_anomaly_score(df, features=list(consensus.keys()))
    df = fe.map_severity(df)
    assert "anomaly_score" in df.columns

  # --- TEST REPORTING (ITALIAN NAMES) ---
  insights = (
    df.group_by("shopping_mission")
    .agg([
      pl.len().alias("receipt_count"),
      pl.col("basket_value").sum().alias("total_mission_revenue"),
      pl.col("basket_value").mean().round(2).alias("avg_trip_value"),
      pl.col("basket_size").mean().round(2).alias("avg_items_per_trip")
    ])
  )
  
  mock_global_insights = [{"period": "2024-06", "data": insights}]
  test_report_path = "reports/test_summary.xlsx"
  os.makedirs("reports", exist_ok=True)

  try:
    generate_global_summary(mock_global_insights, export_path=test_report_path)
    assert os.path.exists(test_report_path)
    
    # Verify Italian Requirements:
    # 1. Month Mapping (06 -> Giugno)
    # 2. Variable Naming (shopping_mission -> Missione_di_Spesa)
    with pd.ExcelFile(test_report_path) as xls:
      assert "Giugno" in xls.sheet_names
      # Check column translation in the specific month sheet
      month_df = pd.read_excel(xls, "Giugno")
      assert "Missione_di_Spesa" in month_df.columns
      assert "Spesa Standard Mista" in month_df["Missione_di_Spesa"].values or \
              "B2B / Ingrosso Outlier" in month_df["Missione_di_Spesa"].values

  finally:
    if os.path.exists(test_report_path):
      os.remove(test_report_path)

  logger.info("✅ Missione Spesa Pipeline Test Passed with Italian Reporting Validation.")