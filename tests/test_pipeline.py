import polars as pl
import logging
import os
from src.engineering import FeatureEngineer

logger = logging.getLogger(__name__)

def test_full_consensus_pipeline():
  """
  Integration Test: Verifies the flow from raw mock data 
  through scoring and automated plotting.
  """
  # Setup Mock Italian Data
  # We use enough rows (6) to allow for 5-fold Lasso CV and VIF checks
  lf = pl.LazyFrame({
    "receipt_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    "Product SKU": ["A", "B", "A", "C", "B", "D", "A", "E", "F", "G", "H", "I"],
    "line_total": [10.0, 5.0, 20.0, 30.0, 5.0, 5.0, 15.0, 15.0, 40.0, 10.0, 5.0, 5.0],
    "Weight": [0.5, None, 1.0, None, None, 0.2, 0.4, None, 1.5, None, None, None],
    "date_str": ["2024-06-01"] * 12,
    "time_str": ["10:00:00"] * 12,
    "Price": [10.0, 5.0, 20.0, 30.0, 5.0, 5.0, 15.0, 15.0, 40.0, 10.0, 5.0, 5.0],
    "Standard_Price": [12.0, 5.0, 18.0, 35.0, 5.0, 6.0, 14.0, 16.0, 45.0, 11.0, 5.0, 5.0]
  })

  # Add derived columns required by the pipeline before the engineer
  lf = lf.with_columns([
    (pl.col("Price") < pl.col("Standard_Price")).alias("is_discounted"),
    pl.when(pl.col("Weight") > 0)
    .then(pl.col("Price") / pl.col("Weight"))
    .otherwise(pl.col("Price"))
    .alias("avg_unit_price")
  ])

  fe = FeatureEngineer(random_state=42)

  # Test Feature Extraction
  processed_df = fe.extract_shopping_mission_features(lf).collect()
  assert "basket_value_per_item" in processed_df.columns
  assert "discount_ratio" in processed_df.columns

  # 3. Test Consensus Selection (The Dictionary Logic)
  features_to_test = ["basket_size", "basket_value_per_item", "discount_ratio"]
  
  # Mocking results to force a consensus for testing
  rf_res = fe.select_features_rf(processed_df, features_to_test, "basket_value", top_n=3)
  ls_res = fe.select_features_lasso(processed_df, features_to_test, "basket_value")
  
  consensus_dict = fe.get_consensus_features(rf_res, ls_res)
  
  # Verify consensus is a dictionary with scores (0-100 range)
  assert isinstance(consensus_dict, dict)
  if consensus_dict:
    first_feat = list(consensus_dict.keys())[0]
    assert 0 <= consensus_dict[first_feat] <= 100

  # 4. Test Anomaly Detection & Scoring
  features_list = list(consensus_dict.keys()) if consensus_dict else features_to_test
  scored_df = fe.add_anomaly_score(processed_df, features=features_list)
  assert "anomaly_score" in scored_df.columns

  # Test Plotting (Ensures KeyError: 0 is fixed)
  # We pass the dictionary to trigger confidence labels
  test_plot_path = "plots/test_anomalies.png"
  
  # Create a 3-feature mock dict if needed to test 3D path
  mock_3d_consensus = {
    "basket_size": 100.0,
    "basket_value_per_item": 30.5,
    "discount_ratio": 5.2
  }
  
  try:
    fe.plot_anomalies(scored_df, features=mock_3d_consensus, path=test_plot_path)
    assert os.path.exists(test_plot_path)
    assert os.path.exists(test_plot_path.replace(".png", "_3d.png"))
  finally:
    # Cleanup
    if os.path.exists(test_plot_path): os.remove(test_plot_path)
    if os.path.exists(test_plot_path.replace(".png", "_3d.png")): 
      os.remove(test_plot_path.replace(".png", "_3d.png"))

  logger.info("✅ Pipeline Smoke Test Passed: Consensus and Plotting are stable.")