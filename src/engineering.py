import polars as pl
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

class FeatureEngineer:
    """Handles feature engineering for sales receipt data."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def calculate_vif(self, df: pl.DataFrame, features: list[str], sample_fraction: float = 0.1) -> pd.DataFrame:
        """ 
        Calculates Variance Inflation Factor (VIF) for selected features. 
        Samples data into memory for statsmodels compatibility.
        
        Args:
            df (pl.DataFrame): Polars DataFrame containing the data.
            features (list[str]): List of column names to calculate VIF for.
            sample_fraction (float): Fraction of data to sample for VIF calculation. 
                                     Used to manage memory for statsmodels.
        Returns:
            pd.DataFrame: A Pandas DataFrame with VIF scores for each feature.
        """
        if not features:
            return pd.DataFrame(columns=['feature', 'vif_factor'])

        print(f"Calculating VIF for {len(features)} features on a {sample_fraction*100}% sample...")

        # Sample the data and convert to Pandas for statsmodels
        # Ensure the sample is collected if the input is a LazyFrame
        if isinstance(df, pl.LazyFrame):
            sampled_df_pl = df.sample(fraction=sample_fraction, seed=self.random_state).collect()
        else:
            sampled_df_pl = df.sample(fraction=sample_fraction, seed=self.random_state)

        if sampled_df_pl.is_empty():
            print("Warning: Sampled DataFrame is empty, cannot calculate VIF.")
            return pd.DataFrame(columns=['feature', 'vif_factor'])

        # Filter for numeric features only for VIF calculation
        numeric_features = [f for f in features if sampled_df_pl[f].dtype in pl.NUMERIC_TYPES]
        if not numeric_features:
            print("No numeric features found for VIF calculation among the provided features.")
            return pd.DataFrame(columns=['feature', 'vif_factor'])

        sampled_df_pd = sampled_df_pl.select(numeric_features).to_pandas()

        # Add a constant to the DataFrame for VIF calculation
        X = add_constant(sampled_df_pd)

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["vif_factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Exclude the constant term from the results
        vif_data = vif_data[vif_data['feature'] != 'const']
        return vif_data.reset_index(drop=True)

    def extract_shopping_mission_features(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """ 
        Extracts 'Shopping Mission' features from the lazy frame.
        Includes Basket Size, Basket Value, and Timestamp features.
        
        Args:
            lf (pl.LazyFrame): The input Polars LazyFrame containing receipt data.
            
        Returns:
            pl.LazyFrame: A new LazyFrame with added shopping mission features.
        """
        print("Extracting shopping mission features...")
        
        # Ensure 'timestamp' column is of datetime type
        lf = lf.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S").alias("timestamp"))

        # Basket Size (number of items per receipt)
        # Basket Value (sum of prices per receipt)
        lf = lf.group_by("receipt_id").agg(
            pl.col("item_id").count().alias("basket_size"),
            pl.col("price").sum().alias("basket_value"),
            pl.col("timestamp").first().alias("receipt_timestamp") # Keep one timestamp per receipt
        ).pipe(lambda df: df.join(lf, on="receipt_id", how="left")) # Join back to original to keep item-level detail

        # Timestamp features
        lf = lf.with_columns(
            pl.col("receipt_timestamp").dt.hour().alias("hour_of_day"),
            pl.col("receipt_timestamp").dt.weekday().alias("day_of_week"), # Monday=1, Sunday=7
            pl.col("receipt_timestamp").dt.week().alias("week_of_year"),
            pl.col("receipt_timestamp").dt.month().alias("month"),
            pl.col("receipt_timestamp").dt.year().alias("year")
        )

        return lf

if __name__ == '__main__':
    # Example usage for testing
    # Create a dummy LazyFrame
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
    lf = pl.LazyFrame(dummy_data)

    feature_engineer = FeatureEngineer()
    
    # Test feature extraction
    lf_features = feature_engineer.extract_shopping_mission_features(lf)
    print("Features extracted (collected for display):")
    print(lf_features.head(5).collect())

    # Test VIF calculation (requires collecting data and numerical features)
    # Ensure the dataframe has numerical columns for VIF
    collected_df = lf_features.collect()
    numerical_cols = [col for col in collected_df.columns if collected_df[col].dtype in pl.NUMERIC_TYPES]
    if 'receipt_id' in numerical_cols: numerical_cols.remove('receipt_id') # Remove ID columns if they are numeric
    if 'item_id' in numerical_cols: numerical_cols.remove('item_id')

    # Add some more numerical features for VIF if they don't exist
    if 'dummy_feature_1' not in collected_df.columns:
        collected_df = collected_df.with_columns(pl.Series("dummy_feature_1", np.random.rand(len(collected_df))) * 10)
    if 'dummy_feature_2' not in collected_df.columns:
        collected_df = collected_df.with_columns(pl.Series("dummy_feature_2", np.random.rand(len(collected_df))) * 5)
    
    # Update numerical_cols after adding dummy features
    numerical_cols = [col for col in collected_df.columns if collected_df[col].dtype in pl.NUMERIC_TYPES]
    if 'receipt_id' in numerical_cols: numerical_cols.remove('receipt_id')
    if 'item_id' in numerical_cols: numerical_cols.remove('item_id')
    if 'price' in numerical_cols: numerical_cols.remove('price') # Often price is the target, not a VIF feature

    print(f"Numerical columns for VIF: {numerical_cols}")

    if numerical_cols:
        vif_results = feature_engineer.calculate_vif(collected_df, numerical_cols)
        print("VIF Results:")
        print(vif_results)
    else:
        print("Not enough numerical columns to calculate VIF.")
