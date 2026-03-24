
from logger.logger import logging
from exception.exception import ProjectError

from entity.config_entity import TransformationConfig
from entity.artifact_entity import ValidationArtifact, TransformationArtifact
from constants.training_pipeline import KNN_IMPUTER_PARAMS, TARGET_COLUMN
from utils.main_utils.utils import write_numpy_array, save_as_pickle

import sys
import os
from pathlib import Path
 
import numpy as np  
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

class TransformationComponent():

    """
        Class for Data Transformation component.
        Transforms the data pulled by the ingestion component if it has been validated by the validation component and imputes any NaN values using KNN imputer.\n
        params ->\n
        ***validation_artifact***: The ValidationArtifact dataclass containing valid train/test file paths and validation status created by validation component.\n
        ***transformation_config***: The TransformationConfig class containing the transformed train/test file and transformation object file and directory paths. 
    """

    def __init__(self, transformation_config: TransformationConfig, validation_artifact: ValidationArtifact):
        
        try:
            logging.info("----- Initializing Data Transformation Component -----")
            
            # initialize config object 
            self.config = transformation_config

            # get train and test file paths from validation artifact
            self.train_file_src_path = validation_artifact.valid_train_file_path
            self.test_file_src_path = validation_artifact.valid_test_file_path
            
            # intialize train/test destination paths
            self.train_x_path = self.config.transformed_trainx_file
            self.train_y_path = self.config.transformed_trainy_file

            self.test_x_path = self.config.transformed_testx_file
            self.test_y_path = self.config.transformed_testy_file

            # initialize imputer destination path
            self.imputer_object_path = self.config.transformation_object_path

        except Exception as e:
            raise ProjectError(e, sys)
        
    @staticmethod
    def read_data(path: str | Path):

        """

        """

        try:
            
            logging.info(f"Reading CSV in location: {path}")
            df: pd.DataFrame = pd.read_csv(path)
            return df

        except Exception as e:
            raise ProjectError(e, sys)        

    def initialize_imputer_object(self) -> Pipeline:

        """"""

        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)

    def initiate_data_transformation(self):

        """"""

        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)