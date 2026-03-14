
# logging and exception imports
from logger.logger import logging
from exception.exception import ProjectError

# validation config and artifact entity import
from entity.config_entity import  ValidationConfig
from entity.artifact_entity import PreprocessingArtifact, ValidationArtifact

# utility function import
from utils.main_utils.utils import read_yaml, write_yaml

# library and function imports
import os
import sys
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.model_selection import train_test_split

class DataValidation:

    """
        Class for Data Validation component.
        Validates the data pulled by the ingestion component and validates it against the base dataset using ks 2 sampling method.\n
        params ->\n
        ***ingestion_artifact***: The IngestionArtifact dataclass containing train/test file paths created by ingestion component\n
        ***validation_config***: The ValidationConfig class containing the valid/invalid and schema directory paths 
    """

    def __init__(self, preprocessing_artifact: PreprocessingArtifact, validation_config: ValidationConfig):
        
        try:
            
            logging.info("----- Initializing Data Validation Component -----")

            self.config = validation_config
            self.schema_path = self.config.schema_file_path

            # read base dataframe
            self.base_dataset_path = self.config.base_dataset_path
            self.base_dataframe = pd.read_csv(self.base_dataset_path)

            # initialize the components and read the contents of schema.yaml
            self.preprocessing_artifact = preprocessing_artifact
            self.preprocessed_file_path = preprocessing_artifact.preprocessed_file_path
            self.processed_dataframe: pd.DataFrame = pd.read_csv(self.preprocessed_file_path)

            logging.info("Reading column_schema.yaml file")

            self.schema = read_yaml(self.schema_path)

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)

        return
    
    def validate_columns(self) -> dict:
        
        """
            Function to validate the columns in the preprocessed dataframe against the columns mentioned in column schema yaml file
            and validation of their data types.\n
            params -> \n
            ***None***\n
            returns ->\n
            ***column_validation***: **dict** containing **2 keys** :\n 1.) if all columns (**numerical and non-numerical**) are present in the pre-processed data frame.\n
             2.) Whether there are any **np.nan** values in any of the numerical columns or not and if there are what columns have those values   

        """
        try:

            # init dict of column validation
            column_validation = {}

            # read column names and their type dict from column schema 
            columns_and_types: dict = self.schema['columns']

            # get set of all columns in schema
            schema_columns = set(columns_and_types.keys())

            # read numerical column names from schema
            schema_numerical_columns: list = self.schema['numerical_columns']

            # get name of all columns in dataframe
            df_columns = set(self.processed_dataframe.columns)

            # check missing columns in dataframe 
            column_validation['missing_columns'] = list(schema_columns - df_columns)

            present_columns = schema_columns & df_columns

            # check for any columns with mismatched dtype or nan values
            column_validation['mismatched_type_columns'] = []
            column_validation['columns_with_nan'] = []

            for column in present_columns:
                if(columns_and_types[column] != str(self.processed_dataframe[column].dtype)):
                    column_validation['mismatched_type_columns'].append(column)

                if(column in schema_numerical_columns and self.processed_dataframe[column].isna().any()):
                    column_validation['columns_with_nan'].append(column)

            return column_validation            

        except Exception as e:
            raise ProjectError(e, sys)
        
    def detect_categorical_drift(self, processed_df: pd.DataFrame, base_df: pd.DataFrame, threshold: float = 0.05) -> dict:

        """
        
        """

        try:

            logging.info("Starting drift detection on categorical columns")
            
            drift_report = {}

            for column in processed_df.columns:

                # get all possible categories in a column
                categories = sorted( 
                                    set(base_df[column].dropna().unique())
                                    .union(
                                        set(processed_df[column].dropna().unique())
                                    ) 
                                )
                
                # skip column with no data or only one unique value (cannot drift)
                if len(categories) < 2:
                    drift_report[column] = {
                        "p_value": 1.0,
                        "degree_of_freedom": int(0),
                        "drift_detected": False,
                    }
                    continue
                
                # get counts of the categories in both the columns
                base_count = base_df[column].value_counts()
                processed_count = processed_df[column].value_counts() 

                # align the two count vectors into equal lengths if not already 
                base_vector = [base_count.get(cat, 0) for cat in categories]
                processed_vector = [processed_count.get(cat, 0) for cat in categories]

                # create contingency table
                contingency_table = [base_vector, processed_vector]

                chi2_score, p_value, dof, expected = chi2_contingency(contingency_table)

                drift_detected = p_value < threshold
            
                drift_report[column] = {
                    "p_value": float(p_value),
                    "degree_of_freedom": int(dof),
                    "drift_detected": drift_detected,
                }

            logging.info("COmpleted categorical drift analysis")

            return drift_report

        except Exception as e:
            raise ProjectError(e, sys)
        
    def detect_numerical_drift(self, df_processed: pd.DataFrame, df_base: pd.DataFrame, threshold: float = 0.05) -> dict:

        """

        """

        try:
            
            logging.info("Starting drift detection on numerical columns")

            drift_report = {}

            for column in df_processed.columns:

                column_base = df_base[column].dropna()
                column_processed = df_processed[column].dropna()

                # safety check to see if columns have zero standard deviation i.e are constant
                std_base = column_base.std()
                std_processed = column_processed.std()

                if(std_base == 0 and std_processed == 0):

                    # check if both columns have equal mean (no drift) or not (drift present but both columns have constant value)
                    mean_base = column_base.mean()
                    mean_processed = column_processed.mean()

                    if(mean_base == mean_processed):
                        drift_report[column] = {
                            'p_value': float(1.0),
                            'mean': float(mean_base),
                            'drift_detected': False
                        }

                    else:
                        drift_report[column] = {
                            'p_value': float(0.0),
                            'mean': float(mean_processed),
                            'drift_detected': True
                        }
                    continue

                # check if both column values come from same distribution
                # statistic near 0 is good (denotes maximum % gap or height difference between two curve heights in the
                # dataframes at the same point), location tells point of maximum drift, positive sign means processed
                # dataset is concentrated towards larger values and negative means towards lower values   
                test_result = ks_2samp(column_base, column_processed, alternative = 'two-sided')

                drift_report[column] = {
                    "p_value": float(getattr(test_result, "pvalue")),
                    "ks_statistic": float(getattr(test_result, "statistic")),
                    "drift_detected": bool(getattr(test_result, "pvalue") < threshold),
                    "direction": getattr(test_result, "statistic_sign"),
                    "location": float(getattr(test_result, "statistic_location"))
                }

            logging.info("Completed numerical drift analysis")

            return drift_report

        except Exception as e:
            raise ProjectError(e, sys)

    def detect_drift(self) -> dict:
        
        """
            Function to detect data drift between the base dataset and preprocessed dataset using ks 2 sampling method and binary/unit columns using chi square test. Data drift is detected both column wise and entire dataframe wise if data is valid the file will be split into train and test files and passed for training and transformation otherwise dumped into invalid data directory.\n
            ***params*** -> \n
            None\n
            ***Returns*** -> \n
            **drift_report**: **dict** containing the drift for each numerical column between the ingested and base dataset.\n
        """

        try:

            logging.info("Intializing drift analysis component")
    
            num_cols = [c for c in self.schema['numerical_columns'] if c in self.processed_dataframe.columns and c != 'sample_id']
            cat_cols = [c for c in self.schema['categorical_columns'] if c in self.processed_dataframe.columns]

            drift_report = {
                'numerical_columns': self.detect_numerical_drift(self.processed_dataframe[num_cols], self.base_dataframe[num_cols]),
                'categorical_columns': self.detect_categorical_drift(self.processed_dataframe[cat_cols], self.base_dataframe[cat_cols])
            }
    
            return drift_report             
        
        except Exception as e:
            raise ProjectError(e, sys)
        
    def validate_data(self, drift_report: dict) -> bool:

        """
        
        """

        try:
            
            # get numerical and categorical drfit reports 
            categorical: dict = drift_report['categorical_columns']
            numerical: dict = drift_report['numerical_columns']

            # get all column names from drift report
            numerical_names: set = numerical.keys() 
            categorical_names: set = categorical.keys()

            all_column_names: set = categorical_names.union(numerical_names)

            # check if all most critical columns are in drfit report and thus not missing from processed dataframe
            critical_columns: set = {'total_normalized_quantity', 'log_normalized_quantity', 'unit_category', 'standardized_unit'}

            validation_status: bool = True

            non_crtical_categories: int = 0
            drifting_categories: int = 0

            if any(s not in all_column_names for s in critical_columns):

                logging.warning(f"Critical Columns {critical_columns - all_column_names} missing from the preprocessed dataset.")

                validation_status = False

                return validation_status

            # check fails i.e all necessary columns are present
            # check for drifts in columns. If any high priority/critical columns have drift discard or else
            # If majority of low priority columns have drift then context has changed and discard

            for column in numerical_names:
                if(numerical[column]['drift_detected'] == True):

                    logging.info(f"Drift detected in {column} column with p-value equal to {numerical[column]['p_value']}. Discarding dataset")

                    validation_status = False
                    return validation_status
                
            if(categorical['standardized_unit']['drift_detected'] == True or categorical['unit_category']['drift_detected'] == True):
                validation_status = False
                return validation_status
            
            for column in categorical_names:
                
                if(categorical[column]['drift_detected'] == True and column not in critical_columns):    
                    drifting_categories += 1
                    non_crtical_categories += 1
                    logging.info(f"Drfit detectd in {column} column of p-value eqaul to {categorical[column]['p_value']}")

            if(drifting_categories >= (int)(non_crtical_categories* 0.5)):
                logging.warning(f"Around {(drifting_categories/non_crtical_categories)*100}% of categorical columns have drifted away. Discarding dataset due to context shift")
                validation_status = False
                return validation_status

            logging.info("Column validation completed no drift detected in high priority columns and little to no drift detected in lower priority columns")

            return validation_status
  
        except Exception as e:
            raise ProjectError(e, sys)
            
    def init_validation(self) -> ValidationArtifact:
        
        """
        
        """
        
        try:
            
            validation_report = {}

            # validate column count
            logging.info("checking for column counts, mismatching data types and columns having NaN values")
            validation_report['column_status'] = self.validate_columns()

            # get drift report
            validation_report['drift_report'] = self.detect_drift()

            # validate drift report 
            validation_status = self.validate_data(validation_report['drift_report'])
            validation_report['validation_status'] = validation_status

            logging.info(f"Writing drift report to {self.config.drift_report_file}")
            write_yaml(validation_report, self.config.drift_report_file)

            if(validation_status):
                logging.info(f"Data validation successful splitting data into train and test file paths and storing inside {self.config.valid_data_dir}")

                train_file, test_file = train_test_split(self.processed_dataframe, self.config.split_ratio)

                train_file.to_csv(self.config.train_file_path, index = False)    

                test_file.to_csv(self.config.test_file_path, index = False)

            else:
                logging.info(f"Validation failed dumping data into {self.config.invalid_data_dir}")
                self.processed_dataframe.to_csv(self.config.invalid_data_dir)

            return ValidationArtifact(validation_status, self.config.train_file_path, self.config.invalid_data_dir, self.config.test_file_path, self.config.drift_report_file)


        except Exception as e:
            raise ProjectError(e, sys)