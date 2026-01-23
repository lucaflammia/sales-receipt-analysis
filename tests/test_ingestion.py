import yaml
from src.ingestion import load_config
import logging

logger = logging.getLogger(__name__)

def test_load_config_local_defaults(tmp_path):
  """
  Test that load_config returns the expected sampling rate for local environment.
  We use a temporary yaml file to isolate the test from your real config.
  """
  # Create a temporary mock config
  d = tmp_path / "subdir"
  d.mkdir()
  config_file = d / "test_config.yaml"

  mock_content = {
    "environment": "local",
    "local": {"sampling_rate": 0.1, "data_path": "data/raw/"},
    "cloud": {"sampling_rate": 1.0, "data_path": "s3://bucket/raw/"},
  }

  with open(config_file, "w") as f:
    yaml.dump(mock_content, f)

  config = load_config(config_path=str(config_file))

  env = config["environment"]
  assert env == "local"
  assert config[env]["sampling_rate"] == 0.1
  assert "data/raw/" in config[env]["data_path"]

  logger.info("✅ test_load_config_local_defaults passed.")

def test_load_config_cloud_switch(tmp_path):
  """Verifies that switching the environment key changes the active sampling rate."""
  config_file = tmp_path / "config.yaml"
  mock_content = {
    "environment": "cloud",
    "local": {"sampling_rate": 0.1},
    "cloud": {"sampling_rate": 1.0},
  }
  with open(config_file, "w") as f:
    yaml.dump(mock_content, f)

  config = load_config(config_path=str(config_file))
  active_env = config["environment"]

  assert active_env == "cloud"
  assert config[active_env]["sampling_rate"] == 1.0

  logger.info("✅ test_load_config_cloud_switch passed.")
