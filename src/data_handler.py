# src/data_handler.py
import pandas as pd


class DataHandler:
    """
    A class to handle loading and initial processing of the energy data.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataHandler with the path to the data file.

        Args:
            file_path (str): The path to the CSV data file.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data, converts 'date' to datetime, and sets it as the index.

        Returns:
            pd.DataFrame: The loaded and pre-processed dataframe.
        """
        try:

            self.df = pd.read_csv(self.file_path, skipinitialspace=True)

            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.set_index("date")
            self.df = self.df.drop(columns=["rv1", "rv2"])
            print("Data loaded and initial processing complete.")
            return self.df
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.file_path}")
            return None
