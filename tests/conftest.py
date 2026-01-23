import os
import pytest
from unittest.mock import patch
import polars as pl
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def setup_testing_env():
  """
    Automatically loads .env and ensures dummy AWS keys exist
    to prevent S3 initialization errors.
  """
  load_dotenv()

  # If keys don't exist in environment, set dummy ones for local testing
  if not os.getenv("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "eu-south-1"

  yield

@pytest.fixture
def mock_s3_pipeline():
  """
    Mocks both the S3 filesystem and the Polars scanning logic.
    Returns the mock objects if you need to perform specific assertions.
  """
  with patch("s3fs.S3FileSystem") as mock_fs, patch("polars.scan_parquet") as mock_scan:

    # Setup default behavior: One file found
    mock_fs.return_value.glob.return_value = ["s3://mock-bucket/data.parquet"]

    # Setup default behavior: Return a valid LazyFrame
    dummy_lf = pl.DataFrame({"id": [1], "val": ["test"]}).lazy()
    mock_scan.return_value = dummy_lf

    yield {"fs": mock_fs, "scan": mock_scan}
