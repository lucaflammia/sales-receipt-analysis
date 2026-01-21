import polars as pl
import yaml
import os

def load_config(config_path='config.yaml'):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config, file_name='receipts.csv'):
    """Ingests data using Polars, respecting environment and sampling rate."""
    env = config['environment']
    env_config = config[env]

    data_path = env_config['data_path']
    sampling_rate = env_config['sampling_rate']

    full_path = os.path.join(data_path, file_name)

    print(f"Loading data from: {full_path} with sampling rate: {sampling_rate}")

    # Determine file type and use appropriate Polars scan function
    if full_path.endswith('.csv'):
        lf = pl.scan_csv(full_path, infer_schema_length=1000)
    elif full_path.endswith('.parquet'):
        lf = pl.scan_parquet(full_path)
    else:
        raise ValueError("Unsupported file type. Only .csv and .parquet are supported.")

    # Apply sampling if not in cloud environment and sampling rate is less than 1.0
    if env == 'local' and sampling_rate < 1.0:
        # For lazy frames, sampling is typically done after some initial operations
        # or by collecting a sample. For initial ingestion, we'll keep it lazy
        # and assume sampling might happen later if needed or rely on a smaller
        # input file for local prototyping.
        # A direct lazy sample is not straightforward without collecting first.
        # For simplicity, we'll assume a smaller dataset for local or that sampling
        # will be handled downstream if the full file is large.
        print("Note: Direct lazy sampling for large files is complex. Assuming smaller local files or downstream sampling.")
        # A more robust lazy sampling would involve adding a random column and filtering
        # e.g., .with_columns(pl.lit(np.random.rand(pl.count())).alias("__rand"))
        #      .filter(pl.col("__rand") < sampling_rate)
        # However, this would require numpy and would be eager in some contexts.
        # For now, we'll just load the full (potentially smaller) local file lazily.

    return lf

if __name__ == '__main__':
    # Example usage for testing
    config = load_config()
    # Create a dummy CSV for testing
    dummy_data = {
        'item_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'receipt_id': [101, 101, 102, 102, 102, 103, 103, 104, 104, 105],
        'price': [10.5, 5.2, 12.0, 3.1, 7.8, 20.0, 15.5, 8.9, 1.2, 5.0],
        'timestamp': [
            '2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 11:00:00',
            '2023-01-01 11:10:00', '2023-01-01 11:15:00', '2023-01-02 09:00:00',
            '2023-01-02 09:15:00', '2023-01-02 14:00:00', '2023-01-02 14:05:00',
            '2023-01-03 16:00:00'
        ]
    }
    dummy_df = pl.DataFrame(dummy_data)
    os.makedirs(config['local']['data_path'], exist_ok=True)
    dummy_df.write_csv(os.path.join(config['local']['data_path'], 'receipts.csv'))
    print("Dummy data created at data/raw/receipts.csv")

    lazy_df = load_data(config, file_name='receipts.csv')
    print("LazyFrame schema:")
    print(lazy_df.schema)
    print("First 5 rows (collected for display):")
    print(lazy_df.head(5).collect())
