# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# library imports
import os
import sys

# artifact and config entity imports
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import TransformationArtifact, ModelTrainerArtifact

# util fucntion imports
from utils.main_utils.utils import save_as_pickle, read_pickle_object, read_numpy_array
from utils.ml_utils.metric.regression_metric import get_prediction_score
from utils.ml_utils.model.estimator.estimator import PredictorModel

# model imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

# grid search function imports
from sklearn.model_selection import GridSearchCV

class ModelTrainer():

    def __init__(self, train_config: ModelTrainerConfig, transformation_artifact: TransformationArtifact):

        try:
            
            logging.info("----- Initializing Model Trainer Object -----")
            self.config = train_config
            self.artifact = transformation_artifact
            self.train_array_path = self.artifact.transformed_train_path
            self.test_array_path = self.artifact.transformed_test_path
            self.preprocessor_path = self.artifact.transformation_object_path 

        except Exception as e:
            raise ProjectError(e, sys)
        
    
