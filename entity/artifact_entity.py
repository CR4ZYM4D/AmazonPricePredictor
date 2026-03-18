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

