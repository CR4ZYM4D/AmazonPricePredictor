
from logger.logger import logging
from exception.exception import ProjectError

from entity.config_entity import TransformationConfig
from entity.artifact_entity import ValidationArtifact, TransformationArtifact
from constants.training_pipeline import KNN_IMPUTER_PARAMS, SIMPLE_IMPUTER_PARAMS, TARGET_COLUMN
from utils.main_utils.utils import write_numpy_array, save_as_pickle

import sys
import os
from pathlib import Path
 
import numpy as np  
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
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
    def read_data(path: str | Path) -> pd.DataFrame:

        """
            Reads the CSV in the specified file path and returns the DataFrame.\n
            params ->\n
            ***path***: Path | String of the CSV to be read.\n
            returns -> \n
            ***df***: The CSV that was stored in the specified file path
        """

        try:
            
            if not (os.path.exists(path)):
                logging.warning(f"There is NO path as {path} to read any CSV check and enter correct path")
                raise ProjectError(f"Invalid path entered to read CSV. No path as {path}", sys)
                                
            else:
                logging.info(f"Reading CSV in location: {path}")
                df: pd.DataFrame = pd.read_csv(path)
                return df

        except Exception as e:
            raise ProjectError(e, sys)        

    def initialize_imputer_object(self) -> Pipeline:

        """
        """

        try:
            
            logging.info("Intializing preprocessor object for column transformation")

            # define continous and categorical features to encode and transform
            continous_features = ['quantity', 'log_normalized_quantity', 'total_normalized_quantity']

            categorical_features = ['unit_category', 'standardized_unit']

            # define numerical pipeline for quantity
            num_pipeline = Pipeline(

                                steps = [

                                    ("imputer", KNNImputer(**KNN_IMPUTER_PARAMS)),
                                    ("scaler", RobustScaler())

                                ]

                           )
            
            # define encoder pipeline for units
            cat_pipeline = Pipeline(

                                steps = [

                                    ("imputer", SimpleImputer(**SIMPLE_IMPUTER_PARAMS)),
                                    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))

                                ]

                           )
            
            # define column transformer object
            preprocessor = ColumnTransformer(

                            transformers = [

                                ('num_pipeline', num_pipeline, continous_features),
                                ('categorical_pipeline', cat_pipeline, categorical_features)

                            ],
                            remainder = 'passthrough'

                           )

            return preprocessor
            
        except Exception as e:
            raise ProjectError(e, sys)

    def initiate_data_transformation(self):

        """
        """

        try:
            
            # read train and test CSV
            train_df: pd.DataFrame = TransformationComponent.read_data(self.train_file_src_path)
            test_df: pd.DataFrame = TransformationComponent.read_data(self.test_file_src_path)

            # drop those columns where target column value is NaN
            train_df = train_df.dropna(subset = [TARGET_COLUMN], inplace= True)
            test_df = test_df.dropna(subset = [TARGET_COLUMN], inplace = True)

            # separate feature and target columns and drop index count column from features if present
            train_y = train_df[TARGET_COLUMN]
            test_y = test_df[TARGET_COLUMN]

            train_x = train_df.drop(columns = [TARGET_COLUMN, 'sample_id'], errors = 'ignore')
            test_x = test_df.drop(columns = [TARGET_COLUMN, 'sample_id'], errors = 'ignore')    



        except Exception as e:
            raise ProjectError(e, sys)