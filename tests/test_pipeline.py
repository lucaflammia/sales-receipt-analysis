import polars as pl
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

def test_polars_to_pandas_handoff():
  """
  Smoke test: Ensures Polars LazyFrames can be collected, 
  converted to Pandas, and fed into Scikit-learn.
  """
  # Create a mock LazyFrame
  lf = pl.LazyFrame({
    "item_id": [1, 2],
    "receipt_id": [101, 101],
    "price": [10.5, 5.0],
    "timestamp": ["2023-01-01 10:00:00", "2023-01-01 10:05:00"]
  })

  # Run a simple engineering step
  # (Assuming extract_features returns a LazyFrame)
  processed_lf = lf.with_columns([
    (pl.col("price") * 1.2).alias("price_with_tax")
  ])

  # Collect and Convert (The "Danger Zone" for memory/types)
  df_pandas = processed_lf.collect().to_pandas()

  # Feed into Model
  model = IsolationForest(n_estimators=10)
  model.fit(df_pandas[["price", "price_with_tax"]])
  
  predictions = model.predict(df_pandas[["price", "price_with_tax"]])

  assert len(predictions) == 2
  assert isinstance(df_pandas, pd.DataFrame)
  logger.info("✅ Smoke test passed: Polars to Pandas handoff is successful.")
  