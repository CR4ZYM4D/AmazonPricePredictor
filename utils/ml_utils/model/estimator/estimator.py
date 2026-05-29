# logging and exception import
from exception.exception import ProjectError
from logger.logger import logging

# library import
import os
import sys
import mlflow.pyfunc

# library imports for type safety 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

class PredictorModel:

    def __init__(self, preprocessor: ColumnTransformer, model):

        try:
            self.preprocessor = preprocessor
            self.model = model
        
        except Exception as e:
            raise ProjectError(e, sys)

    def predict(self, x: pd.DataFrame)-> np.ndarray:

        """
        
        """

        try:

            x_transformed = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transformed)

            return y_pred
        
        except Exception as e:
            raise ProjectError(e, sys)
        

class PredictorWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model: PredictorModel):

        try:
            self.model = model

        except Exception as e:
            raise ProjectError(e, sys)
        
    def predict(self, input: pd.DataFrame) -> np.ndarray:

        try:
            return self.model.predict(input)
        except Exception as e:
            raise ProjectError(e, sys)
