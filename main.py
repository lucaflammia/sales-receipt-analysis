import polars as pl
import logging
import psutil
import time
import os
import numpy as np
from src.ingestion import load_config, load_data
from src.engineering import FeatureEngineer

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_resource_usage(stage: str):
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / (1024 * 1024)
  cpu = process.cpu_percent(interval=None)
  logger.info(f"[{stage}] CPU: {cpu:.2f}% | RAM: {mem:.2f} MB")

def main():
  start_time = time.time()
  logger.info("Starting Sales Receipt Analysis pipeline...")
  log_resource_usage("Start")

  config = load_config()

  env_config = config[config["environment"]]
  file_name = "venduto_dettagli_20240601_20240831_AREA_382-100KROW.csv"

  # --- DYNAMIC SAMPLING LOGIC ---
  sampling_rate = env_config.get("sampling_rate")

  if config["environment"] == "local":
    file_path = os.path.join(env_config["data_path"], file_name)
    if os.path.exists(file_path):
      file_size_gb = os.path.getsize(file_path) / (1024**3)
      if file_size_gb < 1.0:
        logger.info(f"File size ({file_size_gb:.2f} GB) is < 1GB. Setting sampling_rate to 1.0")
        sampling_rate = 1.0
      else:
        sampling_rate = 0.1

  # If sampling_rate is still None (fallback)
  sampling_rate = sampling_rate if sampling_rate is not None else 1.0
  
  # Load Data with Italian Schema mapping
  try:
    raw_data_lazy = load_data(config, file_pattern=file_name)
  except Exception as e:
    logger.info(f"Loading failed: {e}. Using dummy Italian data.")
    raw_data_lazy = pl.LazyFrame(
      {
        "scontrinoIdentificativo": [1, 1, 2, 2, 3],
        "articoloCodice": ["A1", "A2", "A1", "A3", "A2"],
        "totaleLordo": [2.0, 0.5, 1.0, 1.2, 1.2],
        "quantitaPeso": [0.5, None, 0.4, None, None],
        "quantitaPezzi": [1, 1, 1, 1, 2],
        "dataVendita": [
          "2023-01-01",
          "2023-01-01",
          "2023-01-02",
          "2023-01-02",
          "2023-01-03",
        ],
        "oraVendita": [
          "10:00:00",
          "10:00:00",
          "11:00:00",
          "11:00:00",
          "12:00:00",
        ],
      }
    )

  # Apply Sampling if needed
  if sampling_rate < 1.0:
    raw_data_lazy = raw_data_lazy.sample(fraction=sampling_rate)

  raw_data_lazy = raw_data_lazy.rename(
    {
      "scontrinoIdentificativo": "receipt_id",
      "totaleLordo": "line_total",
      "quantitaPeso": "Weight",
      "dataVendita": "date_str",
      "oraVendita": "time_str",
      "articoloCodice": "Product SKU",
    }
  )
  
  # Add Random Price & Standard Price Guardrail
  # We generate a random price and a 'Standard' reference price for testing
  raw_data_lazy = raw_data_lazy.with_columns([
    pl.lit(np.random.uniform(0.5, 50.0)).alias("Price"),
    pl.lit(np.random.uniform(0.5, 50.0)).alias("Standard_Price")
  ])
  
  # Create the row-level Discount Flag
  raw_data_lazy = raw_data_lazy.with_columns(
    (pl.col("Price") < pl.col("Standard_Price")).alias("is_discounted")
  )
  
  raw_data_lazy = raw_data_lazy.with_columns(
    pl.when(pl.col("Weight") > 0)
    .then(pl.col("Price") / pl.col("Weight"))
    .otherwise(pl.col("Price"))
    .alias("avg_unit_price")
  )

  log_resource_usage("Data Ingested & Unit Prices Calculated")

  # Feature Engineering
  feature_engineer = FeatureEngineer()
  processed_data_lazy = feature_engineer.extract_shopping_mission_features(raw_data_lazy)

  logger.info("Materializing data...")    
  eager_processed_data = processed_data_lazy.collect()
  log_resource_usage("Data Materialized")

  # Feature Selection (Consensus)
  features_to_test = [
    "basket_size",
    "basket_value_per_item",
    "avg_basket_unit_price",
    "hour",
    "day_of_week",
    "freshness_weight_ratio",
    "freshness_value_ratio",
    "has_fresh_produce",
    "discount_ratio",
  ]

  logger.info("Running Feature Selection (RF & Lasso)...")
  rf_important = feature_engineer.select_features_rf(
    eager_processed_data, features=features_to_test, target="basket_value", top_n=3
  )
  lasso_important = feature_engineer.select_features_lasso(
    eager_processed_data, features=features_to_test, target="basket_value"
  )
  
  logger.info(f"Full RF Importances: {rf_important}")
  logger.info(f"Full Lasso Coefs: {lasso_important}")

  consensus_features = feature_engineer.get_consensus_features(
    rf_important, lasso_important
  )

  # Validation & Anomalies
  if consensus_features:
    # Check VIF only on chosen features to ensure stability
    vif_results = feature_engineer.calculate_vif(
      eager_processed_data, features=consensus_features
    )
    logger.info(f"Final VIF Results: {vif_results}")

    # Anomaly Detection
    os.makedirs("models", exist_ok=True)
    eager_processed_data = feature_engineer.add_anomaly_score(
      eager_processed_data,
      features=consensus_features,
      model_path="models/iforest.joblib",
    )

    # Visual Verification
    if len(consensus_features) >= 2:
      feature_engineer.plot_anomalies(eager_processed_data, consensus_features)
  else:
    logger.warning(
      "No consensus features found. Using fallback features for anomalies."
    )

  # Save to Parquet
  output_dir = env_config["processed_data_path"]
  os.makedirs(output_dir, exist_ok=True)

  # Define the specific filename
  output_file = os.path.join(output_dir, "processed_receipts.parquet")
  eager_processed_data.write_parquet(output_file)

  logger.info(f"Processed data saved to: {output_file}")
  logger.info(f"Pipeline finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
  main()
