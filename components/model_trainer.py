# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# library imports
import os
import sys
import numpy as np
import mlflow
from dotenv import load_dotenv

# artifact and config entity imports
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import TransformationArtifact, ModelTrainerArtifact

# util fucntion imports
from utils.main_utils.utils import save_as_pickle, read_pickle_object, read_numpy_array, evaluate_models, write_yaml
from utils.ml_utils.metric.flatten_dict import flatten_dict
from utils.ml_utils.metric.regression_metric import get_prediction_score
from utils.ml_utils.model.estimator.estimator import PredictorModel, PredictorWrapper

# model imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

# dagshub import
import dagshub

load_dotenv()

token = os.getenv("DAGSHUB_TOKEN")
if token:
    dagshub.auth.add_app_token(token)

dagshub.init(repo_owner='CR4ZYM4D', repo_name='AmazonPricePredictor', mlflow=True)

class ModelTrainer():

    def __init__(self, transformation_artifact: TransformationArtifact, train_config: ModelTrainerConfig = ModelTrainerConfig()):

        try:
            
            logging.info("----- Initializing Model Trainer Object -----")
            self.config = train_config
            self.artifact = transformation_artifact
            self.train_array_path = self.artifact.transformed_train_path
            self.test_array_path = self.artifact.transformed_test_path
            self.preprocessor_path = self.artifact.transformation_object_path 

        except Exception as e:
            raise ProjectError(e, sys)
        
    def train_model(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> ModelTrainerArtifact:

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

                        "linear_regression": {}, 
                        "k_neighbours_regressor": {
                                                    "n_neighbors": [3, 5, 7], 
                                                    "weights": ["uniform", "distance"],
                                                    "p": [1,2,3]
                                                  },
                        "decision_tree":{
                                          "criterion": ['absolute_error'],
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
                                           "criterion": ['absolute_error'],
                                           "n_jobs": [-1],
                                           "max_features": ["log2", "sqrt", 0.5, 0.7, 0.8]
                                         }
                       }

            # get model performance reports            
            model_report = evaluate_models(models, params, train_x, train_y, test_x, test_y)

            # flatten model report for mlflow
            model_mlflow_report = flatten_dict(model_report)

            # separate hyperparams and metrics from dict for separate logging
            params_to_log = {k: v for k, v in model_mlflow_report.items() if "hyperparameters" in k}
            metrics_to_log = {k: v for k, v in model_mlflow_report.items() if "hyperparameters" not in k}

            # log both metrics and hyperparamters
            with mlflow.start_run(run_name = "training stage", nested = True):
                mlflow.log_metrics(metrics_to_log)
                mlflow.log_params(params_to_log) 

            # sort the dict based on best model to get its artifact
            sorted_keys = sorted(model_report.items(), key = lambda item: item[1]['test_metrics']['mean_absolute_percentage_error'])

            # get best model name
            best_model_name = sorted_keys[0][0]

            # get best model params
            model_params = sorted_keys[0][1]["hyperparameters"]

            # train best model
            best_model = models[best_model_name]
            best_model.set_params(**model_params)

            best_model.fit(train_x, train_y)

            train_y_pred = best_model.predict(train_x)
            test_y_pred = best_model.predict(test_x)

            train_metrics = get_prediction_score(train_y_pred, train_y)
            test_metrics = get_prediction_score(test_y_pred, test_y)

            train_dict: dict = {"train_mean_absolute_percent_error":  train_metrics.mean_absolute_percentage_error,
                                "train_root_mean_squared_error":      train_metrics.root_mean_squared_error,
                                "train_mean_squared_error":           train_metrics.mean_squared_error,
                                "train_r2_score":                     train_metrics.r2_score}
            
            test_dict: dict = { "test_mean_absolute_percent_error":   test_metrics.mean_absolute_percentage_error,
                                "test_root_mean_squared_error":      test_metrics.root_mean_squared_error,
                                "test_mean_squared_error":           test_metrics.mean_squared_error,
                                "test_r2_score":                     test_metrics.r2_score}

            preprocessor = read_pickle_object(self.preprocessor_path)

            predictor_model = PredictorModel(preprocessor, best_model)

            wrapped_model = PredictorWrapper(model = predictor_model)

            with mlflow.start_run():
                
                logging.info("Logging model to MLflow")

                mlflow.pyfunc.log_model(
                    artifact_path=f"artifact_{best_model_name}",
                    python_model=wrapped_model,
                    pip_requirements=["scikit-learn", "pandas", "numpy"] 
                )

                mlflow.log_metrics(train_dict)
                mlflow.log_metrics(test_dict)              

            logging.info(f"saving best model in file path: {self.config.trained_model_path}")
            save_as_pickle(predictor_model, self.config.trained_model_path, overwrite = True)

            logging.info(f"Writing model reports at file path {self.config.reports_path}")
            write_yaml(model_report, self.config.reports_path)

            save_as_pickle(best_model, './final_model/model.pkl')

            return ModelTrainerArtifact(self.config.trained_model_path, train_metrics, test_metrics)

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

            artifact: ModelTrainerArtifact = self.train_model(x_train, x_test, y_train, y_test)

            return artifact

        except Exception as e:
            raise ProjectError(e, sys)