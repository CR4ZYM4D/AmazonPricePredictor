# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# ingestion config and artifact entity import
from entity.config_entity import PreprocessingConfig
from entity.artifact_entity import IngestionArtifact, PreprocessingArtifact

# pre-processing related imports
import emoji
from nltk.corpus import stopwords

# null handling imports
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# library and function imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PreProcessingComponent():

    def __init__(self, ingestion_artifact: IngestionArtifact, preprocessing_config: PreprocessingConfig):
        
        try:
            
            logging.info("----- Initializing Data Preprocessing Component -----")

            # initialize the components and read the contents of ingestion artifact dataclass
            self.ingestion_artifact = ingestion_artifact
            self.ingested_data_path = self.ingestion_artifact.ingested_data_dir

            self.config = preprocessing_config
            self.train_file_path = self.config.training_file_path
            self.test_file_path = self.config.testing_file_path

            # initialize set of stopwords to prevent re initiaslization for every string instance
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.add('\n')

        except ProjectError as e:

            raise(e)
        
    def basic_processing(self, s: str):

        """
            Function for basic proecessing of catalog content column string like converting to lower case, removing emojis and stopwords etc. \n
            params -> \n
            ***s***: string of the catalog_content column of the item\n
            returns -> \n
            s: the processed string with stopwords and emojis removed, in lowercase
        """

        try:
            # convert to lowercase
            s = s.lower()

            # remove emojis
            s = emoji.replace_emoji(s, '')

            # split words
            words = s.split()

            filtered_words = [word for word in words if word not in self.stopwords]

            return ' '.join(filtered_words)

        except ProjectError as e:
            raise(e) 

    def find_quantity_and_unit(self, s: str):

        """
            Function to find the quantity and unit of a product specified in the item catalog_content column\n
            params ->\n
            ***s***: catalog_content string of the item \n 
            returns -> \n
            ***quantity***: 32 bit float amount of quantity\n
            ***unit***: string unit of quantity
        """

        try:
            # make lower case to handle case sensitivity / for safety 
            s= s.lower()

            # find last index of string "value: " 
            value_idx = s.rfind(' value: ')

            # quantity would be as "value: x unit: " because we removed \n 
            unit_idx = s.rfind(" unit: ")

            quantity = np.float32(s[value_idx+8: unit_idx])

            unit = s[unit_idx + 7]

            return (quantity, unit)
        
        except ProjectError as e:
            raise(e)
        
    

    def split_data(self, df: pd.DataFrame):

        """
            Function to split the given Dataframe into train and test subsets as per the config split ratio
            And store the train and test files in the train/test file paths respectively\n
            params ->\n 
            ***df***: Dataframe to be split\n
            returns -> None
        """

        try:
            
            logging.info(f"Splitting Dataframe stored in directory {self.ingested_data_path} in ratio {self.config.split_ratio}")

            # split as per split ratio
            train, test = train_test_split(df, test_size = self.config.split_ratio, random_state=42)

            train_path = self.config.training_file_path
            test_path = self.config.testing_file_path

            if not os.path.exists(train_path):
                logging.info(f"Creating Directory {train_path}") 
                os.makedirs(train_path)

            if not os.path.exists(test_path):
                logging.info(f"Creating Directory {test_path}") 
                os.makedirs(test_path)

            # save in file path mentioned in config
            train.to_csv(train_path)
            test.to_csv(test_path)
            
            logging.info(f"Saved Dataframe split in train and test files into directories {train_path} and {test_path}")
 
        except ProjectError as e:
            raise(e)