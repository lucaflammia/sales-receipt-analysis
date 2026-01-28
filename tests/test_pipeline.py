import polars as pl
import logging
import os
import pytest
from src.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


def test_full_consensus_pipeline():
  """
  Integration Test: Verifies the flow from raw mock data
  through scoring and automated plotting using the Canonical Baseline logic.
  """
  # Setup Mock Italian Data (Normalized format as expected by main.py logic)
  # We use 6 receipts (12 rows) to ensure we have enough points for ML variance
  lf = pl.LazyFrame(
    {
      "receipt_id": [
        "R1",
        "R1",
        "R2",
        "R2",
        "R3",
        "R3",
        "R4",
        "R4",
        "R5",
        "R5",
        "R6",
        "R6",
      ],
      "product_id": ["A", "B", "A", "C", "B", "D", "A", "E", "F", "G", "H", "I"],
      "line_total": [
        10.0,
        5.0,
        20.0,
        30.0,
        5.0,
        5.0,
        15.0,
        15.0,
        40.0,
        10.0,
        5.0,
        5.0,
      ],
      "weight": [0.5, 0.0, 1.0, 0.0, 0.0, 0.2, 0.4, 0.0, 1.5, 0.0, 0.0, 0.0],
      "date_str": ["20240601"] * 12,  # Format YYYYMMDD as expected by str.to_date
      "time_str": ["100000"] * 12,  # Format HHMMSS
      "store_id": ["S_01"] * 12,
      "selfScanning": ["N"] * 12,
    }
  )

  # 2. Simulate Item-Level Enrichment (Logic from main.py)
  # This prepares the data for extract_canonical_features
  lf = lf.with_columns(
    [pl.col("line_total").median().over("product_id").alias("Standard_Price")]
  ).with_columns(
    [
      (pl.col("line_total") < pl.col("Standard_Price"))
      .fill_null(False)
      .alias("is_discounted"),
      (pl.col("line_total") - pl.col("Standard_Price"))
      .fill_null(0.0)
      .alias("price_delta"),
      pl.when(pl.col("weight") > 0)
      .then(pl.col("line_total") / pl.col("weight"))
      .otherwise(pl.col("line_total"))
      .fill_null(0.0)
      .alias("avg_unit_price"),
    ]
  )

  fe = FeatureEngineer(random_state=42)

  # Test Feature Extraction (Method name updated to match engineering.py)
  agg_lf = fe.extract_canonical_features(lf)
  processed_df = agg_lf.collect()

  assert "basket_value_per_item" in processed_df.columns
  assert "discount_ratio" in processed_df.columns
  assert "freshness_weight_ratio" in processed_df.columns
  assert processed_df.height == 6  # Grouped by 6 unique receipt_ids

  # Test Consensus Selection
  features_to_test = [
    "basket_size",
    "basket_value_per_item",
    "freshness_weight_ratio",
  ]

  # Run RF and Lasso importances
  rf_res = fe.select_features_rf(
    processed_df, features_to_test, "basket_value", top_n=3
  )
  ls_res = fe.select_features_lasso(processed_df, features_to_test, "basket_value")

  consensus_dict = fe.get_consensus_features(rf_res, ls_res)

  assert isinstance(consensus_dict, dict)
  if consensus_dict:
    first_feat = list(consensus_dict.keys())[0]
    # Check normalization logic (0-100 range)
    assert 0 <= consensus_dict[first_feat] <= 100

  # Test Anomaly Detection & Scoring
  features_list = list(consensus_dict.keys()) if consensus_dict else features_to_test
  scored_df = fe.add_anomaly_score(processed_df, features=features_list)
  scored_df = fe.map_severity(scored_df)

  assert "anomaly_score" in scored_df.columns
  assert "severity" in scored_df.columns

  # Test Automated Plotting
  test_plot_path = "plots/test_anomalies.png"

  # We use a mock dict of 3 features to ensure both 2D and 3D plotting paths are hit
  mock_consensus = {
    "basket_size": 100.0,
    "basket_value_per_item": 50.0,
    "freshness_weight_ratio": 25.0,
  }

  try:
    fe.plot_anomalies(scored_df, features=mock_consensus, path=test_plot_path)

    # Verify both 2D pairwise and 3D "Money Shot" plots were generated
    assert os.path.exists(test_plot_path), "2D plots not generated"
    assert os.path.exists(
        test_plot_path.replace(".png", "_3d.png")
    ), "3D plot not generated"

  finally:
    # Cleanup
    for suffix in ["", "_3d.png"]:
      p = (
        test_plot_path
        if suffix == ""
        else test_plot_path.replace(".png", suffix)
      )
      if os.path.exists(p):
        os.remove(p)

  logger.info("✅ Pipeline Integration Test Passed.")
