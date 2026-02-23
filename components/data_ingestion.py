
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
from dotenv import load_dotenv 

import sys

# load env variables and get DB URL
load_dotenv()
MONGO_DB_URL = os.getenv("DB_URL")

class IngestionComponent():

    """
        Class for the Data Ingestion Component.
        Ingests data from the mongo db collection and splits it into train and test files storing them within the feature store\n
        params ->\n
        ***ingestion_config***: IngestionConfig class object containing artifact, ingestion, ingested and feature_store directory paths
        along with DB and collection anme and split ratio for train/test files
    """

    def __init__(self, ingestion_config: IngestionConfig):
        
        try:
            
            self.config = ingestion_config
            logging.info("----- Initializing Data Ingestion Component -----")

        except Exception as e:
            raise ProjectError(e, sys)
        
    def export_collection_as_df(self):

        """
            Function to connect to the Database using MongoClient and get the required collection\n 
            params -> None\n
            returns -> Collection stated in config class converted into a Dataframe
        """

        try:
            
            # get DB and collection name from config
            db_name = self.config.db_name
            collection_name = self.config.collection_name

            # connect to DB using DB URL
            logging.info(f"Attempting connecting to Database {db_name} and extracting collection {collection_name}")
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            # get collection from DB
            collection = self.mongo_client[db_name][collection_name]

            # convert into pandas DF
            logging.info("Connection successful. Extracting collection and converting to dataframe")
            df = pd.DataFrame(list(collection.find()))

            # remove _id column if made by default
            if '_id' in df.columns.to_list():
                df.drop(labels = "_id", axis = 1, inplace = True)
            
            # replace null values
            df.replace({'na': np.nan}, inplace = True)
            return df

        except Exception as e:
            raise ProjectError(e, sys)
        
    def export_to_ingested_data(self, df: pd.DataFrame):

        """
            Function to get ingested data path from config class and stores the Dataframe  
            csv in the ingested data directory. Creates the ingested data directory if it does not already exist\n
            params ->\n 
            ***df***: Dataframe to be saved\n
            returns -> None
        """

        try:
            
            path = self.config.ingested_dir
            
            if not os.path.exists(path):
                logging.info(f"Creating Directory {path}") 
                os.makedirs(path)

            # save collection as a csv for training/testing
            df.to_csv(self.config.ingested_dir)
            logging.info(f"Saved Dataframe as CSV in directory {path}")

            return

        except Exception as e:
            raise ProjectError(e, sys)
            
    def initiate_ingestion(self) -> IngestionArtifact:

        """
            Function to intake the data from the Collection, convert it into a Dataframe, 
            and save the files as a csv in the ingested directory\n
            params -> None\n
            returns -> IngestionArtifact Dataclass object containing ingested file paths
        """

        try:
            # get DB collection as a dataframe
            df: pd.DataFrame = self.export_collection_as_df()

            self.export_to_ingested_data(df)

            return IngestionArtifact(self.config.ingested_dir)

        except Exception as e:
            raise ProjectError(e, sys)