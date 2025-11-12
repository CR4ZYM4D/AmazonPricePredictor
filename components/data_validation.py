
# logging and exception imports
from logger.logger import logging
from exception.exception import ProjectError

# validation config and artifact entity import 
from entity.config_entity import  ValidationConfig
from entity.artifact_entity import IngestionArtifact, ValidationArtifact

# utility function import
from utils.main_utils.utils import read_yaml

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

    def __init__(self, ingestion_artifact: IngestionArtifact, validation_config: ValidationConfig):
        
        try:
            
            logging.info("-----Initializing Data Validation Component-----")

            # initialize the components and read the contents of schema.yaml
            self.ingestion_artifact = ingestion_artifact
            self.config = validation_config
            self.schema_path = self.config.schema_file_path

            logging.info("Reading schema.yaml file")

            self.schema = read_yaml(self.schema)

        except ProjectError as e:
            raise(e)

        return
    
    def validate_columns(self):
        
        """
            Function to validate the columns in the ingested dataframe agains the columns mentioned in schema.yaml file
            and validation their data types
        """
        try:
            pass
        except ProjectError as e:
            raise(e)

    def detect_drift(self):
        
        """
            Function to detect data drift between the base model dataset and passed dataset using ks 2 sampling method 
        """

        try:
            pass
        except ProjectError as e:
            raise(e)
        

    
    def init_validation(self) -> ValidationArtifact:
        
        try:
            pass
        except ProjectError as e:
            raise(e)