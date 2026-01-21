import polars as pl
import yaml
import psutil
import time
import os

from src.ingestion import load_config, load_data
from src.engineering import FeatureEngineer

# Placeholder for ML models
from sklearn.ensemble import IsolationForest
# from mlxtend.frequent_patterns import fpgrowth
# import ollama

def log_resource_usage(stage):
    """Logs current RAM and CPU usage."""
    pid = os.getpid()
    process = psutil.Process(pid)
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
    print(f"[{stage}] RAM Usage: {ram_usage_mb:.2f} MB, CPU Usage: {cpu_percent:.2f}%")

def main():
    print("Starting sales-receipt-analysis pipeline...")
    log_resource_usage("Start")

    # 1. Load Configuration
    config = load_config()
    env = config['environment']
    env_config = config[env]
    random_state = config['random_state']

    print(f"Running in {env} environment.")

    # 2. Data Ingestion (Polars LazyFrame)
    print("\n--- Data Ingestion ---")
    lazy_df = load_data(config, file_name='receipts.csv') # Assuming 'receipts.csv' is present
    print("Data loaded as Polars LazyFrame.")
    log_resource_usage("After Ingestion (Lazy)")

    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")
    feature_engineer = FeatureEngineer(random_state=random_state)
    processed_lazy_df = feature_engineer.extract_shopping_mission_features(lazy_df)
    print("Shopping mission features extracted (Lazy).")
    log_resource_usage("After Feature Engineering (Lazy)")

    # 4. Collect data for VIF and ML (this is where computation happens)
    print("\n--- Data Collection & VIF Calculation ---")
    print(f"Collecting data with sampling rate: {env_config['sampling_rate']}")
    # Apply sampling here if it wasn't handled during lazy loading, or if we need a smaller
    # collected DF for memory-intensive operations like VIF or initial ML model training.
    if env_config['sampling_rate'] < 1.0:
        collected_df = processed_lazy_df.sample(fraction=env_config['sampling_rate'], seed=random_state).collect()
    else:
        collected_df = processed_lazy_df.collect()
    
    print(f"Collected data shape: {collected_df.shape}")
    log_resource_usage("After Data Collection (Eager)")

    # Identify numerical features for VIF calculation
    numerical_features = [col for col in collected_df.columns 
                          if collected_df[col].dtype in pl.NUMERIC_TYPES 
                          and col not in ['receipt_id', 'item_id', 'timestamp', 'receipt_timestamp']]
    
    if numerical_features:
        vif_results = feature_engineer.calculate_vif(collected_df, numerical_features, sample_fraction=0.1) # VIF itself samples again
        print("\nVariance Inflation Factor (VIF) Results:")
        print(vif_results)
    else:
        print("No suitable numerical features for VIF calculation.")
    log_resource_usage("After VIF Calculation")

    # 5. Anomaly Detection (IsolationForest example)
    print("\n--- Anomaly Detection (IsolationForest) ---")
    # For demonstration, let's use some of the numerical features for anomaly detection
    ad_features = ['basket_size', 'basket_value', 'hour_of_day', 'day_of_week']
    ad_features = [f for f in ad_features if f in collected_df.columns and collected_df[f].dtype in pl.NUMERIC_TYPES]

    if ad_features:
        print(f"Training IsolationForest on features: {ad_features}")
        X = collected_df.select(ad_features).to_pandas()
        model = IsolationForest(random_state=random_state)
        collected_df = collected_df.with_columns(pl.Series("anomaly_score", model.fit_predict(X).flatten()))
        print("Anomaly detection complete. Added 'anomaly_score' column.")
        print("First 5 rows with anomaly scores:")
        print(collected_df.head(5))
    else:
        print("Not enough suitable features for Anomaly Detection.")
    log_resource_usage("After Anomaly Detection")

    # 6. Save processed data (optional, for cloud environment or next steps)
    processed_output_path = env_config['processed_data_path']
    output_file_name = "processed_receipts.parquet"
    final_output_path = os.path.join(processed_output_path, output_file_name)
    
    # Ensure local processed data directory exists
    if env == 'local':
        os.makedirs(processed_output_path, exist_ok=True)
        collected_df.write_parquet(final_output_path)
        print(f"\nProcessed data saved to: {final_output_path}")
    else:
        print(f"\nIn cloud environment, processed data would be saved to: {final_output_path} (S3 upload logic needed)")

    log_resource_usage("End")
    print("\nSales receipt analysis pipeline finished.")

if __name__ == '__main__':
    main()
