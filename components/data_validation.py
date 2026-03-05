
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
        Validates the data pulled by the ingestion component and validates it against the base dataset using ks 2 sampling method.\n
        params ->\n
        ***ingestion_artifact***: The IngestionArtifact dataclass containing train/test file paths created by ingestion component\n
        ***validation_config***: The ValidationConfig class containing the valid/invalid and schema directory paths 
    """

    def __init__(self, preprocessing_artifact: PreprocessingArtifact, validation_config: ValidationConfig):
        
        try:
            
            logging.info("----- Initializing Data Validation Component -----")

            # read base dataframe
            self.base_dataset_path = self.config.base_dataset_path
            self.base_dataframe = pd.read_csv(self.base_dataset_path)

            # initialize the components and read the contents of schema.yaml
            self.preprocessing_artifact = preprocessing_artifact
            self.preprocessed_file_path = preprocessing_artifact.preprocessed_file_path
            self.processed_dataframe: pd.DataFrame = pd.read_csv(self.preprocessed_file_path)

            self.config = validation_config
            self.schema_path = self.config.schema_file_path

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
            Function to detect data drift between the base dataset and preprocessed dataset using ks 2 sampling method and binary/unit columns using chi square test. Data drift is detected both column wise and entire dataframe wise if data is valid the file will be split into train and test files and passed for training and transformation otherwise dumped into invalid data directory.\n
            ***params*** -> \n
            None\n
            ***Returns*** -> \n
            **drift_report**: **dict** containing the drift for each numerical column between the ingested and base dataset.\n
            **validation_status**: **bool** if the number of numerical columns drift is too much it deems the dataset as invalid and unfit for usage
        """

        try:

            # get numerical column names
            numerical_columns = list(set(self.schema['numerical_columns']) & self.processed_dataframe.columns)

            # get categorical column names
            categorical_columns = list(set(self.schema['categorical_columns']) & self.processed_dataframe.columns)

            # drop sample id column as it is pointless to sample against
            df_processed = df_processed.drop('sample_id', axis = 1) 
            df_base = df_base.drop('sample_id', axis = 1)

            # get the numerical and categorical features in two different dataframes
            df_processed_numerical = df_processed[numerical_columns]
            df_base_numerical = df_base[numerical_columns]

            df_processed_categorical = df_processed[categorical_columns]
            df_base_categorical = df_base[categorical_columns]

            # sample between the two
            self.detect_numerical_drift(df_processed_numerical, df_base_numerical)

            self.detect_categorical_drift(df_processed_categorical, df_base_categorical)              

        except Exception as e:
            raise ProjectError(e, sys)
            
    def init_validation(self) -> ValidationArtifact:
        
        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)