# logging and error import
from logger.logger import logging
from exception.exception import ProjectError

# library import
import os
import sys
from pathlib import Path
from typing import Any

# library imports for non OS/sys things 
import yaml
import numpy as np
import pickle
from dataclasses import asdict
from sklearn.model_selection import GridSearchCV

# fucntion and classes imports
from utils.ml_utils.metric.regression_metric import get_prediction_score

def read_yaml(file_path: Path | str) -> dict: 

    """
        Reads The YAML file passed in the specified path and returns the stream object\n 
        params -> \n
        ***file_path*** : Path | str containing the path of the yaml file that needs to be read \n
        returns ->\n YAML file stream object
    """

    try:
        
        logging.info(f"Reading {file_path} file")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding = 'utf-8') as f:

                schema = yaml.safe_load(f)

                if schema is None:
                    logging.warning("Schema is empty! no data has been read")
                    raise ProjectError("Schema is empty! no data has been read", sys)

                logging.info(f"Successfully read the yaml file")
                return schema
        else: 
            raise ProjectError(f"The specified YAML File Path {file_path} does not exist! Please ensure the path is correct and try again", sys)

    except Exception as e:
        raise ProjectError(e, sys)
    
def write_yaml(data: dict, file_path: Path | str, overwrite: bool = False):

    """
        Writes a data stream into the specified file path as a yaml file. If the path already exists, it throws an error and stops execution by default.\n
        Can be set to overwrite the pre-existing file contents as well.\n
        params ->\n
        ***data***: The data to be written into the yaml file. \n
        ***file_path***: Path | str The path of the yaml file the data is to to be written in. \n
        ***overwrite***: bool *default* = False Whether to overwrite the file if it already exists. False by default. \n
        returns -> \n
        None
    """

    try: 
        
        if os.path.exists(file_path) and overwrite == False:

            logging.info(f"The specified file path {file_path} exists and overwrite was set to False. Exitting loop")
            raise(ProjectError("The specified file already exists!", sys))
        
        logging.info(f"Creating file {file_path}")

        folder_path = file_path[ : file_path.rfind('/') + 1]
        os.makedirs(folder_path, exist_ok = True)
        
        with open(file_path, 'w', encoding = 'utf-8') as f:

            yaml.dump(data, f, default_flow_style= False)
            logging.info(f"Successfully written data into file {file_path}")
        return

    except Exception as e:
        raise ProjectError(e, sys)
    
def read_numpy_array(file_path: Path | str) -> np.ndarray:
    
    """
        Reads The numpy array passed in the specified path and returns the array object\n 
        params -> \n
        ***file_path*** : Path | str containing the path of the numpy array that needs to be read \n
        returns ->\n numpy arrray object
    """
    
    try:
            
        logging.info(f"Reading {file_path} file")
        
        if os.path.exists(file_path):
            
            array: np.ndarray = np.load(file_path)

            if array is None:
                logging.warning("array is empty! no data has been read")
                raise ProjectError("array is empty! no data has been read", sys)

            logging.info(f"Successfully read the numpy array")
            return array
        else: 
            raise ProjectError(f"The specified nunmpy array {file_path} does not exist! Please ensure the path is correct and try again", sys)

    except Exception as e:
        raise ProjectError(e, sys)
    
def write_numpy_array(array: np.ndarray, file_path: Path | str, overwrite: bool = False):

    """
        Writes a numpy array into the specified file path as a numpy array. If the path already exists, it throws an error and stops execution by default.\n
        Can be set to overwrite the pre-existing file contents as well.\n
        params ->\n
        ***data***: The data to be written into the array. \n
        ***file_path***: Path | str The path of the file the array is to to be written in. \n
        ***overwrite***: bool *default* = False Whether to overwrite the file if it already exists. False by default. \n
        returns -> \n
        None
    """

    try: 
        
        if os.path.exists(file_path) and overwrite == False:

            logging.info(f"The specified file path exists {file_path} and overwrite was set to False. Exitting loop")
            raise(ProjectError("The specified file already exists!", sys))
        
        logging.info(f"Creating file {file_path}")

        folder_path = file_path[ : file_path.rfind('/') + 1]
        os.makedirs(folder_path, exist_ok = True)
        
        with open(file_path, 'wb') as f:

            np.save(f, array, allow_pickle = False)
            logging.info(f"Successfully written data into file {file_path}")
        return

    except Exception as e:
        raise ProjectError(e, sys)
    
def read_pickle_object(file_path: Path | str): 

    """
        Reads The pickle object passed in the specified path and returns the object\n 
        params -> \n
        ***file_path*** : Path | str containing the path of the numpy array that needs to be read \n
        returns ->\n pickle object
    """

    try:

        logging.info(f"Reading {file_path} file")
        
        if os.path.exists(file_path):
            
            with open(file_path, 'rb') as f:
                object = pickle.load(f)

                if object is None:
                    logging.warning("pickled object is empty! no data has been read")
                    raise ProjectError("pickled object is empty! no data has been read", sys)

                logging.info(f"Successfully read the pickle object")
                return object
        else: 
            raise ProjectError(f"The specified pickle object {file_path} does not exist! Please ensure the path is correct and try again", sys)
        
    except Exception as e:
        raise ProjectError(e, sys) 
    
def save_as_pickle(object: Any, file_path: Path | str, overwrite: bool = False):
    
    """
        Writes data into the specified file path as a pickle object. If the path already exists, it throws an error and stops execution by default.\n
        Can be set to overwrite the pre-existing file contents as well.\n
        params ->\n
        ***data***: The data to be written into the object. \n
        ***file_path***: Path | str The path of the file the array is to to be written in. \n
        ***overwrite***: bool *default* = False Whether to overwrite the file if it already exists. False by default. \n
        returns -> \n
        None
    """

    try: 
        
        if os.path.exists(file_path) and overwrite == False:

            logging.info(f"The specified file path {file_path} exists and overwrite was set to False. Exitting loop")
            raise(ProjectError("The specified file already exists!", sys))
        
        logging.info(f"Creating file {file_path}")
        
        folder_path = file_path[ : file_path.rfind('/') + 1]
        os.makedirs(folder_path, exist_ok = True)
        
        with open(file_path, 'wb') as f:

            pickle.dump(object, f)
            logging.info(f"Successfully written data into file {file_path}")
        return

    except Exception as e:
        raise ProjectError(e, sys)
    
def evaluate_models(models: dict, params: dict, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict:

    """
    
    """

    try:
        
        # init dict to return
        report = {}

        # init dict to track metrics
        score_dict = {
                        "rmse": "neg_root_mean_squared_error",
                        "mape": "neg_mean_absolute_percentage_error",
                        "mse": "neg_mean_squared_error",
                        "R2": 'r2' 
                     }

        # iterate through all models and params
        for key, value in models.items():

            # get params dict
            param = params.get(key)

            # init grid search choosing model with best MAPE
            grid = GridSearchCV(value, param, cv = 5, scoring = score_dict, refit = "mape", n_jobs = -1)

            # fit on x and y train
            grid.fit(train_x, train_y)

            # get best model of its type
            model = grid.best_estimator_

            # get prediction on train and test x to compare and put in report
            train_y_pred = model.predict(train_x)
            test_y_pred = model.predict(test_x)

            # get metrics comparing prediction and actual values
            train_metrics: dict = asdict(get_prediction_score(train_y_pred, train_y))
            test_metrics: dict = asdict(get_prediction_score(test_y_pred, test_y))

            # log metrics inside model key
            report[key] = {"train_metrics": train_metrics, "test_metrics": test_metrics, "hyperparameters": grid.best_params_}
            

        # return reportand best models
        return report
      
    except Exception as e:
        raise ProjectError(e, sys)