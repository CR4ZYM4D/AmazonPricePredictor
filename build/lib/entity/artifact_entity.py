from dataclasses import dataclass

@dataclass
class IngestionArtifact:
    """
        Dataclass for the ingestion artifact containing the ingested data file paths    
    """
    ingested_data_dir: str


@dataclass
class PreprocessingArtifact:
    """
        Dataclass for the ingestion artifact containing the preprocessed data file paths    
    """
    preprocessed_file_path: str
    

@dataclass
class ValidationArtifact:
    """
        Dataclass for the validation artifact containing the valid/invalid train and test file paths. Validation status and the drift report path    
    """
    
    validation_status: bool
    valid_train_file_path: str
    invalid_train_file_path: str
    valid_test_file_path: str
    drift_report_path:str


@dataclass
class TransformationArtifact:
    """
        Dataclass for the transformation artifact containing the transfomred train and test file paths along with the transformation object file path.    
    """
    transformed_train_path: str
    transformed_test_path: str
    transformation_object_path: str        

@dataclass
class MetricArtifact:
    """
        Dataclass containing the metrics of the models i.e. R^2 variance, MAE, MSE and RMSE
        for easy access and storage 
    """
    r2_score: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float

@dataclass
class ModelTrainerArtifact:
    """
        Dataclass for the model trainer artifact containing the trained model path and the model metrics artifact for train and test datasets  
    """

    model_file_path: str
    train_metric_artifact: MetricArtifact
    test_metric_artifact: MetricArtifact