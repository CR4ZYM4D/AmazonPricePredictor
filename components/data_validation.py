
# logging and exception imports
from logger.logger import logging
from exception.exception import ProjectError

# validation config and artifact entity import
from entity.config_entity import  ValidationConfig
from entity.artifact_entity import PreprocessingArtifact, ValidationArtifact

# utility function import
from utils.main_utils.utils import read_yaml, write_yaml

# library and function imports
import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation:

    """
        Class for Data Validation component.
        Validates the data pulled by the ingestion component and tests it against the null hypothesis using ks 2 sampling method.\n
        params ->\n
        ***ingestion_artifact***: The IngestionArtifact dataclass containing train/test file paths created by ingestion component\n
        ***validation_config***: The ValidationConfig class containing the valid/invalid and schema directory paths 
    """

    def __init__(self, preprocessing_artifact: PreprocessingArtifact, validation_config: ValidationConfig):
        
        try:
            
            logging.info("----- Initializing Data Validation Component -----")

            # initialize the components and read the contents of schema.yaml
            self.preprocessing_artifact = preprocessing_artifact
            self.preprocessed_file_path = preprocessing_artifact.preprocessed_file_path
            self.config = validation_config
            self.schema_path = self.config.schema_file_path

            self.processed_dataframe: pd.DataFrame = pd.read_csv(self.preprocessed_file_path)

            logging.info("Reading column_schema.yaml file")

            self.schema = read_yaml(self.schema_path)

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)

        return
    
    def validate_columns(self):
        
        """
            Function to validate the columns in the preprocessed dataframe against the columns mentioned in column schema yaml file
            and validation of their data types.\n
            params -> \n
            ***None***\n
            returns ->\n
            ***column_validation***: **dict** containing **2 keys** :\n 1.) if all columns (**numerical and non-numerical**) are present in the pre-processed data frame.\n
             2.) Whether there are any **np.nan** values in any of the numerical columns or not and if there are what columns have those values   

        """
        try:

            # init dict of column validation
            column_validation = {}

            # read column names and their type dict from column schema 
            columns_and_types: dict = self.schema['columns']

            # get set of all columns in schema
            schema_columns = set(columns_and_types.keys())

            # read numerical column names from schema
            schema_numerical_columns: list = self.schema['numerical_columns']

            # get name of all columns in dataframe
            df_columns = set(self.processed_dataframe.columns)

            # check missing columns in dataframe 
            column_validation['missing_columns'] = list(schema_columns - df_columns)

            present_columns = schema_columns & df_columns

            # check for any columns with mismatched dtype or nan values
            column_validation['mismatched_type_columns'] = []
            column_validation['columns_with_nan'] = []

            for column in present_columns:
                if(columns_and_types[column] != str(self.processed_dataframe[column].dtype)):
                    column_validation['mismatched_type_columns'].append(column)

                if(column in schema_numerical_columns and self.processed_dataframe[column].isna().any()):
                    column_validation['columns_with_nan'].append(column)

            return column_validation            

        except Exception as e:
            raise ProjectError(e, sys)

    def detect_drift(self):
        
        """
            Function to detect data drift between the base model dataset and passed dataset using ks 2 sampling method 
        """

        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)
            
    def init_validation(self) -> ValidationArtifact:
        
        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)