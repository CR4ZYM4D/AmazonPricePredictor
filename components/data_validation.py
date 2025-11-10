
# logging and exception imports
from logger.logger import logging
from exception.exception import ProjectError

# validation config and artifact entity import 
from entity.config_entity import  ValidationConfig
from entity.artifact_entity import IngestionArtifact, ValidationArtifact

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
        return
