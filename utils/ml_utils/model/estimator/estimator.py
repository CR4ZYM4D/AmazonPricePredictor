# logging and exception import
from exception.exception import ProjectError
from logger.logger import logging

# library import
import os
import sys

# library imports for type safety 
import numpy as np
from sklearn.compose import ColumnTransformer

# model name and directory import
from constants.training_pipeline import MODEL_DIR_NAME, MODEL_NAME

class PredictorModel:

    def __init__(self, preprocessor: ColumnTransformer, model):

        try:
            self.preprocessor = preprocessor
            self.model = model
        
        except Exception as e:
            raise ProjectError(e, sys)

    def predict(self, x: np.ndarray):

        """
        
        """

        try:

            x_transformed = self.preprocessor.transform(x)
            y_pred = self.model.predict(x)

            return y_pred
        
        except Exception as e:
            raise ProjectError(e, sys)

