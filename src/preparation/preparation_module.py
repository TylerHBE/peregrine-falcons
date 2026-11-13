from abc import ABC, abstractmethod
import os
import pandas as pd
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()
data = os.getenv("DATA_PATH", "../data") 

# Preparation module base class for data preparation
class PreparationModule(ABC):

    def __init__(self, name: str):
        self.name = name

    def get_csv(self, name: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        :param name: The name of the CSV file (without extension).
        :return: A pandas DataFrame containing the CSV data.
        """
        return pd.read_csv(f"{data}/{name}.csv")

    @abstractmethod
    def prepare(self, rerun=False) -> pd.DataFrame:
        pass

    @abstractmethod
    def training_set(self) -> pd.DataFrame:
        """
        Return the training set from the provided DataFrame.

        -> return: A pandas DataFrame containing the training set.
        """
        pass

    @abstractmethod
    def test_set(self) -> pd.DataFrame:
        """
        Return the test set from the provided DataFrame.

        -> return: A pandas DataFrame containing the test set.
        """
        pass

    @abstractmethod
    def tensor_to_df(self, tensor) -> pd.DataFrame:
        """
        Convert the output tensor to a pandas DataFrame.

        (tensor): The output tensor to be converted.
        -> return: A pandas DataFrame containing the converted data.
        """
        pass

    @abstractmethod
    def display_df(self, df: pd.DataFrame) -> str:
        """
        Display the contents of a pandas DataFrame as an html visualization.

        (df: pd.DataFrame): The pandas DataFrame to be displayed.
        -> return: An html visualization of the DataFrame.
        """
        pass