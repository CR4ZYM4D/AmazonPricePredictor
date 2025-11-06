
# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# ingestion config and artifact entity import
from entity.config_entity import IngestionConfig
from entity.artifact_entity import IngestionArtifact

# library and function imports
import os
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv 

# load env variables and get DB URL
load_dotenv()
MONGO_DB_URL = os.getenv("DB_URL")

"""
    Class for the Data Ingestion Component
    params -> ingestion_config: IngestionConfig class object 
"""

class IngestionComponent():

    def __init__(self, ingestion_config: IngestionConfig):
        
        try:
            
            self.config = ingestion_config
            logging.info("----- Initializing Data Ingestion Component -----")

        except ProjectError as e:
            raise(e)
        
    def initiate_ingestion(self):

        """
            Function to intake the data from the Collection, convert it into a Dataframe, split it into train-test ratio 
            and save the train/test files as a csv in the ingested directory
            params -> None
            returns -> IngestionArtifact Dataclass object containing train and test file paths
        """

        try:
            # get DB collection as a dataframe
            df: pd.DataFrame = self.export_collection_as_df()

            # save collection as a csv for training/testing
            df.to_csv(self.config.ingested_dir)

            logging.info(f"Saved Dataframe as CSV in directory {self.config.ingested_dir}")
            logging.info(f"Splitting Dataframe stored in directory {self.config.ingested_dir} in ratio {self.config.split_ratio}")

            # get x (base features) and y(price) in separate dataframes
            y = df['price']
            x = df.drop('price', axis = 1, inplace=True)

            # split as per split ratio
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = self.config.split_ratio, random_state=42)

            # save in file path mentioned in config
            x_train.to_csv(os.path.join(self.config.training_file_path, 'x'))
            y_train.to_csv(os.path.join(self.config.training_file_path, 'y'))
            x_test.to_csv(os.path.join(self.config.testing_file_path, 'x'))
            y_test.to_csv(os.path.join(self.config.testing_file_path, 'y'))

            logging.info(f"Split Dataframe in train and test x and y files in directories {self.config.training_file_path} and {self.config.testing_file_path}")

            return IngestionArtifact(self.config.training_file_path, self.config.testing_file_path)

        except ProjectError as e:
            raise(e)
        
    def export_collection_as_df(self):

        """

        """

        try:
            pass
        except ProjectError as e:
            raise(e)