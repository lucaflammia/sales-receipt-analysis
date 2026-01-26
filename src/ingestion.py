import polars as pl
import yaml
import os
import glob
import s3fs
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
  """Loads the configuration from a YAML file."""
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  # Validation
  load_dotenv()
  key = os.getenv("AWS_ACCESS_KEY_ID", "NOT_SET")
  if key == "your_access_key_here" or key == "NOT_SET":
    logger.warning("⚠️ Warning: No real AWS credentials found. Using placeholders.")
  return config


def get_s3_fs():
  load_dotenv()

  # Explicitly fetch from environment
  key = os.getenv("AWS_ACCESS_KEY_ID")
  secret = os.getenv("AWS_SECRET_ACCESS_KEY")
  token = os.getenv("AWS_SESSION_TOKEN")
  region = os.getenv("AWS_DEFAULT_REGION", "eu-south-1")

  if not key or not secret:
    raise EnvironmentError(
      "Missing AWS Credentials. Check GitHub Secrets or .env file."
    )

  return s3fs.S3FileSystem(
    key=key,
    secret=secret,
    token=token,
    client_kwargs={"region_name": region},
    config_kwargs={
      "signature_version": "s3v4"
    },  # Often required for newer SSO roles
  )

def validate_source(path, is_cloud=False):
  """
    Checks if the data source exists and is not empty.
    Returns the list of files found.
  """
  if is_cloud:
    fs = get_s3_fs()
    # glob returns a list of paths
    files = fs.glob(path)
  else:
    # Local glob
    files = glob.glob(path)

  if not files:
    raise FileNotFoundError(f"⚠️ Validation Failed: No files found at {path}")

  logger.info(f"✅ Source Validated: Found {len(files)} file(s).")
  return files

def load_data(config, file_pattern=None):
  """
    Ingests data using Polars Lazy API.
    Supports both S3 (cloud) and local environments for CSV files.
  """
  env = config["environment"]
  env_config = config[env]
  data_path = env_config["data_path"]

  # Ensure the path ends with a slash for clean joining
  if not data_path.endswith("/"):
    data_path += "/"

  # Build the full search path
  search_path = f"{data_path}{file_pattern}"

  if env == "cloud":
    logger.info(f"☁️ Attempting Cloud Ingestion: {search_path}")

    pq_pattern = (
      file_pattern.replace(".csv", ".parquet") if file_pattern else "*.parquet"
    )
    pq_search_path = f"{data_path}{pq_pattern}"
    storage_options={"file_system": get_s3_fs()}

    try:
      logger.info(f"🔍 Checking for Parquet files: {pq_search_path}")
      validate_source(pq_search_path, is_cloud=True)

      logger.info(f"✨ Found Parquet! Scanning: {pq_search_path}")
      return pl.scan_parquet(
        pq_search_path, storage_options=storage_options
      )

    except (FileNotFoundError, Exception) as e:
      logger.warning(
        f"⚠️ Parquet not found or inaccessible: {e}. Falling back to CSV..."
      )

      # --- Fallback to CSV ---
      csv_search_path = pq_search_path.replace(".parquet", ".csv")

      fs = get_s3_fs()
      files = fs.glob(csv_search_path)

      if not files:
        raise FileNotFoundError(f"No CSV files found at {csv_search_path}")

      target_file = files[0]
      logger.info(f"📄 Opening S3 Stream for: {target_file}")

      # Use the fs object to open the file as a stream.
      # This bypasses Polars' attempt to look at the local disk.
      # We use read_csv(...).lazy() because scan_csv requires a path string.
      with fs.open(target_file, mode="rb") as f:
        return pl.read_csv(
          f,
          separator=";",
          infer_schema_length=1000,
          ignore_errors=True,
          rechunk=True,
        ).lazy()

  else:
    logger.info(f"🏠 Attempting Local Ingestion: {search_path}")

    # Validate Local files
    validate_source(search_path, is_cloud=False)

    # Lazy Scan Local CSV
    return pl.scan_csv(search_path, infer_schema_length=1000)

if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )

  config = load_config()

  # Example execution
  try:
    # For cloud testing, we might need a specific folder
    file_pattern = (
      "receipts.csv" if config["environment"] == "local" else "venduto_testata/*.parquet"
    )
    lf = load_data(config, file_pattern=file_pattern)

    # Preview the first few rows (collecting triggers the authenticated download)
    df_preview = lf.head(5).collect()
    logger.info(f"Successfully loaded data preview:\n{df_preview}")

  except Exception as e:
    logger.error(f"🚨 Pipeline Error: {e}")
