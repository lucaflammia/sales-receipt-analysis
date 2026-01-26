import logging
import polars as pl
import pytest
from src.ingestion import load_data

logger = logging.getLogger(__name__)


def test_load_data_cloud_structure(mock_s3_pipeline, monkeypatch):
  """
  Tests cloud path logic by mocking the S3 filesystem.
  """
  # Setup Mock Config
  config = {
    "environment": "cloud",
    "cloud": {"data_path": "s3://mock-bucket/", "region": "eu-south-1"},
  }

  # Mock the S3 data returned by the filesystem
  # We simulate that glob finds one file
  mock_fs = mock_s3_pipeline["fs"].return_value
  mock_fs.glob.return_value = ["s3://mock-bucket/test_data.parquet"]

  # Use monkeypatch to return a dummy DataFrame when Polars scans the path
  # This prevents Polars from trying to actually read the 's3://' bytes
  dummy_df = pl.DataFrame({"id": [1], "product": ["Mock Item"], "price": [10.5]})

  def mock_scan_parquet(*args, **kwargs):
    return dummy_df.lazy()

  monkeypatch.setattr(pl, "scan_parquet", mock_scan_parquet)

  # Run the ingestion
  lf = load_data(config, file_pattern="*.parquet")
  df = lf.collect()

  assert not df.is_empty()
  assert df.columns == ["id", "product", "price"]

  # Verify the filesystem was interacted with correctly
  mock_fs.glob.assert_called_with("s3://mock-bucket/*.parquet")

  logger.info(f"✅ Cloud structure test passed. Glob path verified.")
