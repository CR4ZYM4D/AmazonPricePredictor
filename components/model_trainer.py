# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# library imports
import os
import sys
import numpy as np

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
        
    def train_model(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):

        """

        """

        try: 
            
            # init models
            models: dict = {
                        "linear_regression": LinearRegression(),
                        "k_neighbours_regressor": KNeighborsRegressor(),
                        "decision_tree": DecisionTreeRegressor(),
                        "ada_boost": AdaBoostRegressor(),
                        "gradient_boost": GradientBoostingRegressor(),
                        "random_forest": RandomForestRegressor()  
                    }
            
            # init model params
            params: dict = {
                        "k_neighbours_regressor": {
                                                    "n_neighbours": [3, 5, 7], 
                                                    "weights": ["uniform, distance"],
                                                    "p": [1,2,3]
                                                  },
                        "decision_tree":{
                                          "criterion": 'absolute_error',
                                          "splitter": ["best", "random"],
                                          "max_features": ["log2", "sqrt", 0.5, 0.7, 0.8]
                                        },
                        "ada_boost": {
                                      "learning_rate": [0.25, 0.5, 1, 1.5, 2], 
                                      "loss": ['linear', 'exponential', 'square']
                                     },
                        "gradient_boost": {
                                           "loss": ['squared_error', 'huber', 'quantile', "absolute_error"],
                                           "learning_rate": [0.25, 0.5, 1, 1.5, 2],
                                           "n_estimators": [50, 75, 100, 125, 150],
                                          },
                        "random_forest": {
                                           "n_estimators": [50, 75, 100, 125, 150],
                                           "criterion": 'absolute_error',
                                           "n_jobs": -1,
                                           "max_features": ["log2", "sqrt", 0.5, 0.7, 0.8]
                                         }
                       }
            
            

        except Exception as e:
            raise ProjectError(e, sys)
        
    def initiate_model_training(self) -> ModelTrainerArtifact:

        """

        """

        try:
            
            # read train testr arrays
            transformed_train_array = read_numpy_array(self.train_array_path)
            transformed_test_array = read_numpy_array(self.test_array_path)

            # split train/test arrays into x & y
            x_train = transformed_train_array[:, :-1]
            x_test = transformed_train_array[:, -1]

            y_train = transformed_test_array[:, :-1]
            y_test = transformed_test_array[:, -1]

            self.train_model(x_train, x_test, y_train, y_test)



        except Exception as e:
            raise ProjectError(e, sys)