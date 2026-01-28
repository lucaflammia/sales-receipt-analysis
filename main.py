import polars as pl
import logging
import psutil
import time
import os
import argparse
import glob
from src.ingestion import load_config
from src.engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_resource_usage(stage: str):
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / (1024 * 1024)
  logger.info(f"[{stage}] RAM: {mem:.2f} MB")

def find_partition_path(area_root, year, month):
  pattern = os.path.join(area_root, "**", f"year={year}", f"month={month}")
  matches = glob.glob(pattern, recursive=True)
  dirs = [m for m in matches if os.path.isdir(m)]
  return dirs[0] if dirs else None

def process_partition(year, month, config, feature_engineer, days=None):
  env_config = config[config["environment"]]
  area_code = env_config.get('area_code', '382')
  
  area_path = os.path.join(
    env_config["data_path"], 
    "raw_normalized", "venduto", f"area={area_code}"
  )

  base_data_path = find_partition_path(area_path, year, month)
  
  if not base_data_path:
    logger.warning(f"⚠️ Path not found for Year={year}, Month={month}")
    return

  logger.info(f"📂 Found Data: {base_data_path}")

  all_files = []
  if days:
    for d in days:
      day_glob = os.path.join(base_data_path, f"day={d}", "*.parquet")
      all_files.extend(glob.glob(day_glob))
  else:
    full_glob = os.path.join(base_data_path, "day=*", "*.parquet")
    all_files = glob.glob(full_glob)

  if not all_files:
    logger.warning(f"⚠️ No parquet files found in {base_data_path}")
    return

  logger.info(f"📊 Found {len(all_files)} files to scan.")

  try:
    # Load LazyFrame
    raw_data_lazy = pl.scan_parquet(all_files)
    
    # Schema Normalization
    raw_data_lazy = raw_data_lazy.with_columns([
      pl.col("scontrinoIdentificativo").cast(pl.String),
      pl.col("articoloCodice").cast(pl.String)
    ]).rename({
      "scontrinoIdentificativo": "receipt_id",
      "totaleLordo": "line_total",
      "quantitaPeso": "weight",
      "negozioCodice": "store_id",
      "scontrinoData": "date_str",
      "scontrinoOra": "time_str",
      "articoloCodice": "product_id"
    })

    # Item-Level Enrichment
    raw_data_lazy = raw_data_lazy.with_columns([
      pl.lit(f"{year}-{month:02d}").alias("partition_key"),
      pl.col("line_total").median().over("product_id").alias("Standard_Price")
    ]).with_columns([
      (pl.col("line_total") < pl.col("Standard_Price")).fill_null(False).alias("is_discounted"),
      (pl.col("line_total") - pl.col("Standard_Price")).fill_null(0.0).alias("price_delta"),
      pl.when(pl.col("weight") > 0).then(pl.col("line_total") / pl.col("weight"))
        .otherwise(pl.col("line_total")).fill_null(0.0).alias("avg_unit_price")
    ])

    # Feature Extraction & Aggregation
    df_lazy = feature_engineer.extract_canonical_features(raw_data_lazy)
    df = df_lazy.collect()

    if df.height == 0:
      logger.error(f"🛑 Empty DataFrame for {year}-{month} after aggregation.")
      return

    # ML Preparation
    features_to_test = [
      "basket_size", "basket_value_per_item", "hour_of_day", 
      "is_weekend", "store_numeric", "freshness_weight_ratio", "avg_price_delta_per_basket"
    ]
    available_features = [f for f in features_to_test if f in df.columns]
    
    # Diagnostic Log
    null_stats = df.select(available_features).null_count()
    logger.info(f"🕵️ Feature Null Counts: {null_stats.to_dicts()[0]}")

    # Fill Nulls/NaNs to ensure ML processing
    df = df.with_columns([
      pl.col(available_features).fill_null(0.0).fill_nan(0.0)
    ])

    logger.info(f"✅ Aggregated {df.height} baskets. Starting ML...")

    # Anomaly Detection & Plotting
    df_ml = df.drop_nulls(subset=["basket_value"])

    if df_ml.height < 10:
      logger.warning(f"⚠️ Not enough data for ML in {year}-{month} (Rows: {df_ml.height}).")
    else:
      # Feature Importance
      rf_imp = feature_engineer.select_features_rf(df_ml, features=available_features, target="basket_value")
      lasso_imp = feature_engineer.select_features_lasso(df_ml, features=available_features, target="basket_value")
      consensus = feature_engineer.get_consensus_features(rf_imp, lasso_imp)

      if consensus:
        suffix = f"d{min(days)}_{max(days)}" if days else "full"
        model_path = f"models/iforest_{year}_{month}_{suffix}.joblib"
        
        # Scoring
        df = feature_engineer.add_anomaly_score(df, features=consensus, model_path=model_path)
        df = feature_engineer.map_severity(df)

        # --- Plotting Step ---
        plot_filename = f"anomalies_{year}_{month}_{suffix}.png"
        plot_path = os.path.join("plots", plot_filename)
        feature_engineer.plot_anomalies(df, features=rf_imp, path=plot_path)

    # Export
    export_root = os.path.join(env_config["processed_data_path"], f"area={area_code}", f"year={year}", f"month={month}")
    if days:
      export_root = os.path.join(export_root, f"days_{min(days)}_to_{max(days)}")
        
    os.makedirs(export_root, exist_ok=True)
    df.write_parquet(os.path.join(export_root, "canonical_baseline.parquet"))
    logger.info(f"✅ Saved to: {export_root}")

  except Exception as e:
    logger.error(f"❌ Processing Error for {year}-{month}: {e}")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--year", type=int, required=True)
  parser.add_argument("--month", type=int, required=True)
  parser.add_argument("--month_end", type=int)
  parser.add_argument("--day", type=int)
  parser.add_argument("--day_end", type=int)
  args = parser.parse_args()

  start_time = time.time()
  config = load_config()
  fe = FeatureEngineer()
  
  months = range(args.month, (args.month_end or args.month) + 1)
  target_days = list(range(args.day, (args.day_end or args.day) + 1)) if args.day else None
  
  for m in months:
    process_partition(args.year, m, config, fe, days=target_days)
    log_resource_usage(f"Completed Month {m}")

  logger.info(f"⏱️ Total Execution Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
  main()
