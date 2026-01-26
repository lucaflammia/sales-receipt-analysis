import os
import time
import logging
import pytest
from src.ingestion import load_config, load_data

logger = logging.getLogger(__name__)

pytest.mark.skipif(
  os.getenv("AWS_SESSION_TOKEN") is None,
  reason="S3 integration test skipped: No AWS_SESSION_TOKEN found in environment"
)
def test_s3_tree_connection():
  config = load_config()
  # Ensure environment is set to cloud for this test
  config["environment"] = "cloud"
  target_folders = ["venduto_dettagli/"]
  logger.info("🚀 Starting S3 Latency Audit...")

  for folder in target_folders:
    start_time = time.perf_counter()
    try:
      # We look for parquet files inside these specific folders
      # Note: If the folders contain CSVs, change the pattern in load_data
      path = f"{folder}*.parquet" 
      lf = load_data(config, file_pattern=path)

      schema = lf.schema

      end_time = time.perf_counter()
      duration = end_time - start_time

      logger.info(
        f"✅ {folder:20} | Time: {duration:.2f}s | Columns: {len(schema)}"
      )

    except Exception as e:
      logger.error(
        f"❌{folder:20} | Failed after {time.perf_counter() - start_time:.2f}s | Error: {str(e)[:250]}..."
      )
      # This makes pytest turn RED
      pytest.fail(f"S3 Connection failed for {folder}")
