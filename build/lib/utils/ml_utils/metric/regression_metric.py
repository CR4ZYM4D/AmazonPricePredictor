# exception class import
from exception.exception import ProjectError

# library import
import sys

# metric artifact class import
from entity.artifact_entity import MetricArtifact 

# numpy import
import numpy as np

# metric function imports
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error

def get_prediction_score(y_predicted: np.ndarray, y_true: np.ndarray) -> MetricArtifact:

    """
        
    """

    try: 
        
        # get r^2 variance between y_true and y_pred
        var = r2_score(y_true, y_predicted)

        # get MSE
        mse = mean_squared_error(y_true, y_predicted)

        # get MAPE
        mape = mean_absolute_percentage_error(y_true, y_predicted)

        # get RMSE
        rmse = root_mean_squared_error(y_true, y_predicted)

        return MetricArtifact(var, mse, rmse, mape)

    except Exception as e:
        raise ProjectError(e, sys)