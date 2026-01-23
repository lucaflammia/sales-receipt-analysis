import logging
from src.ingestion import load_data

logger = logging.getLogger(__name__)

def test_load_data_cloud_structure(mock_s3_pipeline):
  config = {
    "environment": "cloud",
    "cloud": {"data_path": "s3://mock-bucket/", "region": "eu-south-1"}
  }

  lf = load_data(config, file_pattern="*.parquet")
  df = lf.collect()

  assert not df.is_empty()

  # Verify the glob was called with the right S3 path
  mock_s3_pipeline["fs"].return_value.glob.assert_called_once()
  logger.info(f"Called glob with: {mock_s3_pipeline['fs'].return_value.glob.call_args}")
  logger.info("✅ test_load_data_cloud_structure passed.")
