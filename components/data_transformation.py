
from logger.logger import logging
from exception.exception import ProjectError

from entity.config_entity import TransformationConfig
from entity.artifact_entity import ValidationArtifact, TransformationArtifact
from constants.training_pipeline import KNN_IMPUTER_PARAMS, TARGET_COLUMN
from utils.main_utils.utils import write_numpy_array, save_as_pickle

import sys
import os
 
import numpy as np  
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

class TransformationComponent():

    def __init__(self, transformation_config: TransformationConfig, validation_artifact: ValidationArtifact):
        
        """"""

        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)
        
    @staticmethod
    def read_data():

        """"""

        try:
            pass
        except Exception as e:
            raise ProjectError(e, sys)        

    def initialize_imputer_object(transformation_object, pipeline_object: Pipeline):

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