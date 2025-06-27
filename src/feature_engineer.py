# src/feature_engineer.py
import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    A class to perform all feature engineering tasks on the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the FeatureEngineer with a dataframe.

        Args:
            df (pd.DataFrame): The input dataframe to engineer features on.
        """
        self.df = df.copy()

    def create_time_features(self):
        """Creates basic time-based features."""
        self.df["hour"] = self.df.index.hour
        self.df["day_of_week"] = self.df.index.dayofweek
        self.df["day_of_month"] = self.df.index.day
        self.df["month"] = self.df.index.month
        return self

    def create_cyclical_features(self):
        """Creates cyclical features for time-based attributes."""
        self.df["hour_sin"] = np.sin(self.df["hour"] * (2 * np.pi / 24))
        self.df["hour_cos"] = np.cos(self.df["hour"] * (2 * np.pi / 24))
        self.df["day_of_week_sin"] = np.sin(self.df["day_of_week"] * (2 * np.pi / 7))
        self.df["day_of_week_cos"] = np.cos(self.df["day_of_week"] * (2 * np.pi / 7))
        return self

    def create_lag_and_rolling_features(self):
        """Creates lag and rolling window features for the target variable."""
        target = "Appliances"
        self.df[f"{target}_lag_1hr"] = self.df[target].shift(6)
        self.df[f"{target}_lag_24hr"] = self.df[target].shift(144)
        self.df[f"{target}_rolling_mean_6"] = (
            self.df[target].rolling(window=6).mean().shift(1)
        )
        self.df[f"{target}_rolling_std_6"] = (
            self.df[target].rolling(window=6).std().shift(1)
        )
        return self

    def create_lag_and_rolling_features(self):
        """Creates a comprehensive set of lag and rolling window features."""
        target = "Appliances"

        # Define time horizons (in 10-minute steps)
        # 1 hour, 3 hours, 6 hours, 12 hours, 24 hours
        lags = [6, 18, 36, 72, 144]

        for lag in lags:
            self.df[f"{target}_lag_{lag}"] = self.df[target].shift(lag)

        # Rolling window features over different periods
        # Using 1-hour and 24-hour windows
        rolling_windows = [6, 144]
        for window in rolling_windows:
            self.df[f"{target}_rolling_mean_{window}"] = (
                self.df[target].rolling(window=window).mean().shift(1)
            )
            self.df[f"{target}_rolling_std_{window}"] = (
                self.df[target].rolling(window=window).std().shift(1)
            )
            self.df[f"{target}_rolling_min_{window}"] = (
                self.df[target].rolling(window=window).min().shift(1)
            )
            self.df[f"{target}_rolling_max_{window}"] = (
                self.df[target].rolling(window=window).max().shift(1)
            )

        return self

    def engineer_features(self) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.

        Returns:
            pd.DataFrame: The dataframe with all engineered features.
        """
        self.create_time_features()
        self.create_cyclical_features()
        self.create_lag_and_rolling_features()
        self.df = self.df.dropna()
        print("Full feature engineering pipeline complete.")
        return self.df
