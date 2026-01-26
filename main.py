import polars as pl
import logging
import psutil
import time
import os
import numpy as np
from src.ingestion import load_config, load_data
from src.engineering import FeatureEngineer

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

  # Add Random Price Guardrail & Map Columns
  # We generate a random price between 0.5 and 50.0 for testing
  raw_data_lazy = raw_data_lazy.with_columns(
    [pl.lit(np.random.uniform(0.5, 50.0)).alias("Price")]
  )

  # Rename columns for consistency
  # raw_data_lazy = raw_data_lazy.rename({"dataVendita": "Date", "quantitaPezzi": "Amount", "oraVendita": "Time", "scontrinoIdentificativo": "receipt_id", "barcodeCodice": "Product SKU"})

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

  log_resource_usage("Data Ingested & Price Generated")

  # Feature Engineering
  feature_engineer = FeatureEngineer()
  processed_data_lazy = feature_engineer.extract_shopping_mission_features(raw_data_lazy)

  # VIF Calculation
  vif_features = [
    "basket_value",
    "basket_size",
    "hour",
    "has_fresh_produce",
    "freshness_weight_ratio",
    "freshness_value_ratio",
  ]

  logger.info("Materializing data for VIF...")
  eager_processed_data = processed_data_lazy.collect()
  log_resource_usage("Data Collected")
  
  os.makedirs("models", exist_ok=True)

  vif_results = feature_engineer.calculate_vif(
    eager_processed_data,
    features=vif_features,
    sample_fraction=env_config["sampling_rate"],
  )
  logger.info("VIF Results:", vif_results)

  # Anomaly Detection
  anomaly_features = ["basket_value", "basket_size"]
  eager_processed_data = feature_engineer.add_anomaly_score(
    eager_processed_data,
    features=anomaly_features,
    model_path="models/iforest.joblib",
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
