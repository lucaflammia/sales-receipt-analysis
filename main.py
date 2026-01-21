import pandas as pd
import numpy as np
import polars as pl
import yaml
import psutil
import time
import os
from datetime import datetime, timedelta

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

def generate_dummy_data(file_path='data/raw/receipts.csv', num_receipts=200):
    """
    Generates item-level receipt data. 
    Each receipt contains multiple items to allow for 'Basket Size' aggregations.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    np.random.seed(42)
    
    item_ids = []
    receipt_ids = []
    prices = []
    timestamps = []

    # Start date for timestamps
    base_time = datetime(2023, 1, 1, 10, 0, 0)

    item_counter = 1
    for r_id in range(101, 101 + num_receipts):
        # Determine how many items are in this specific receipt (Basket Size)
        # Normal distribution around 5 items, minimum of 1
        basket_size = max(1, np.random.poisson(5))
        
        # Random time for this receipt (spread over 3 days)
        receipt_time = base_time + timedelta(
            days=np.random.randint(0, 3),
            hours=np.random.randint(0, 8),
            minutes=np.random.randint(0, 60)
        )

        for _ in range(basket_size):
            item_ids.append(item_counter)
            receipt_ids.append(r_id)
            # Normal price distribution between 1.0 and 25.0
            prices.append(round(np.random.uniform(0.5, 25.0), 2))
            # Items in the same receipt have timestamps within minutes of each other
            item_time = receipt_time + timedelta(minutes=np.random.randint(0, 15))
            timestamps.append(item_time.strftime('%Y-%m-%d %H:%M:%S'))
            
            item_counter += 1

    # Create Dictionary
    data = {
        'item_id': item_ids,
        'receipt_id': receipt_ids,
        'price': prices,
        'timestamp': timestamps
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Inject a few specific anomalies (extremely high prices for single items)
    # This helps validate your IsolationForest logic later
    df.loc[0:2, 'price'] = 499.99 

    # Save to CSV
    df.to_csv(file_path, index=False)

    print(f"✅ Created '{file_path}' with {len(df)} item records across {num_receipts} receipts.")
    print("\nSample of generated data:")
    print(df.head(10))

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
    # Check if receipts.csv exists, if not generate dummy data
    data_path = os.path.join(env_config['data_path'], 'receipts.csv')
    if not os.path.exists(data_path):
        generate_dummy_data(file_path=data_path, num_receipts=1000)
    lazy_df = load_data(config, file_name='receipts.csv')
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
        collected_df = processed_lazy_df.collect().sample(fraction=env_config['sampling_rate'], seed=random_state)
    else:
        collected_df = processed_lazy_df.collect()
    
    print(f"Collected data shape: {collected_df.shape}")
    log_resource_usage("After Data Collection (Eager)")

    # Identify numerical features for VIF calculation
    numerical_features = [col for col in collected_df.columns 
                          if collected_df[col].dtype in pl.NUMERIC_DTYPES 
                          and col not in ['receipt_id', 'item_id', 'timestamp', 'receipt_timestamp']]
    
    if numerical_features:
        vif_results = feature_engineer.calculate_vif(collected_df, numerical_features, sample_fraction=0.1)
        print("\nVariance Inflation Factor (VIF) Results:")
        print(vif_results)
    else:
        print("No suitable numerical features for VIF calculation.")
    log_resource_usage("After VIF Calculation")

    # 5. Anomaly Detection (IsolationForest example)
    print("\n--- Anomaly Detection (IsolationForest) ---")
    # For demonstration, let's use some of the numerical features for anomaly detection
    ad_features = ['basket_size', 'basket_value', 'hour_of_day', 'day_of_week']
    ad_features = [f for f in ad_features if f in collected_df.columns and collected_df[f].dtype in pl.NUMERIC_DTYPES]

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
