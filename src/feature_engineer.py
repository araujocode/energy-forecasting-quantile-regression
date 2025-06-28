import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    A class to perform a focused and robust set of feature engineering tasks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the FeatureEngineer with a dataframe.
        """
        self.df = df.copy()

    def create_time_features(self):
        """Creates basic time-based features."""
        self.df["hour"] = self.df.index.hour
        self.df["day_of_week"] = self.df.index.dayofweek
        self.df["month"] = self.df.index.month
        return self

    def create_cyclical_features(self):
        """Creates cyclical features for time-based attributes."""
        self.df["hour_sin"] = np.sin(self.df["hour"] * (2 * np.pi / 24))
        self.df["hour_cos"] = np.cos(self.df["hour"] * (2 * np.pi / 24))
        return self

    def create_lag_and_rolling_features(self):
        """Creates lag and rolling features based on the original target scale."""
        target = "Appliances"

        # A lag for weekly seasonality
        self.df[f"{target}_lag_24hr"] = self.df[target].shift(144)

        # A rolling window for the recent trend (1 hour)
        self.df[f"{target}_rolling_mean_1hr"] = (
            self.df[target].rolling(window=6).mean().shift(1)
        )
        self.df[f"{target}_rolling_std_1hr"] = (
            self.df[target].rolling(window=6).std().shift(1)
        )

        return self

    def transform_target(self, target_col="Appliances"):
        """Applies a log1p transformation to the target variable."""
        print("Applying log1p transformation to the target variable.")
        self.df[target_col] = np.log1p(self.df[target_col])
        return self

    def engineer_features(self) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.
        """
        self.create_time_features()
        self.create_cyclical_features()
        self.create_lag_and_rolling_features()
        self.df = (
            self.df.dropna()
        )  # Drop NaNs from lag/rolling before transforming target
        self.transform_target()  # Transform the target variable last
        print("Curated feature engineering pipeline complete.")
        return self.df
