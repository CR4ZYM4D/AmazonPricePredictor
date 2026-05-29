# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# ingestion config and artifact entity import
from entity.config_entity import PreprocessingConfig
from entity.artifact_entity import IngestionArtifact, PreprocessingArtifact

# pre-processing related imports
import emoji
from nltk.corpus import stopwords

# library and function imports
import os
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from utils.main_utils.utils import read_yaml

import sys

# conversion constants for a single measuring unit of each type 
FLOZ_TO_ML = 29.5735
L_TO_ML = 1000
OZ_TO_G = 28.3495
LB_TO_G = 453.592
KG_TO_G = 1000


class PreProcessingComponent():

    """
        Class for Data Preprocessing component.
        Pre-processes and performs feature engineering on the data pulled by the ingestion component and stores it in the preprocessing directory path.\n
        params ->\n
        ***ingestion_artifact***: The IngestionArtifact dataclass containing ingested file path created by ingestion component\n
        ***preprocessing_config***: The PreprocessingConfig class containing the Feature store and preprocessed file path. 
    """    

    def __init__(self, ingestion_artifact: IngestionArtifact, preprocessing_config: PreprocessingConfig = PreprocessingConfig()) -> None:
        
        try:
            
            logging.info("----- Initializing Data Preprocessing Component -----")

            # initialize the components and read the contents of ingestion artifact dataclass
            self.ingestion_artifact = ingestion_artifact
            self.ingested_data_path = self.ingestion_artifact.ingested_data_dir

            self.config = preprocessing_config
            self.processed_file_directory = self.config.preprocessing_directory
            self.preprocessed_file_path = self.config.preprocessed_file_path

            self.schema_file_path = self.config.schema_file_path

            # initialize set of stopwords to prevent re initiaslization for every string instance
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.add('\n')

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)
        
    def basic_processing(self, s: str) -> str:

        """
            Function for basic proecessing of catalog content column string like converting to lower case, removing emojis and stopwords etc. \n
            params -> \n
            ***s***: string of the catalog_content column of the item\n
            returns -> \n
            ***s***: the processed string with stopwords and emojis removed, in lowercase
        """

        try:
            # convert to lowercase
            s = s.lower()

            # remove emojis
            s = emoji.replace_emoji(s, '')

            # split words
            words = s.split()

            filtered_words = [word for word in words if word not in self.stopwords]

            return ' '.join(filtered_words)

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys) 

    def find_quantity_and_unit(self, s: str) -> tuple:

        """
            Function to find the quantity and unit of a product specified in the item catalog_content column\n
            params ->\n
            ***s***: catalog_content string of the item \n 
            returns -> \n
            ***quantity***: 32 bit float amount of quantity\n
            ***unit***: string unit of quantity
        """

        try:
            # make lower case to handle case sensitivity / for safety 
            s= s.lower()

            # find last index of string "value: " 
            value_idx = s.rfind(' value: ')

            # quantity would be as "value: x unit: " because we removed \n 
            unit_idx = s.rfind(" unit: ")

            if unit_idx == -1:
                unit_idx = s.rfind(' unit')

            quantity = np.float32(s[value_idx+7: unit_idx])

            unit = s[unit_idx + 6: ].strip() if unit_idx != -1 else 'ambiguous'

            return (quantity, unit)
        
        except Exception as e:
            logging.error(e)
            return ProjectError(e, sys)
        
    def standardize_unit(self, df: pd.DataFrame, column_name: str = 'unit') -> pd.DataFrame:

        """
            function to standardize the units present in the specified column of the dataframe such as converting "fl_oz" and "fl oz" to "ml"\n
            params ->\n
            ***df***: dataframe whose units are to be standardized\n
            ***column_name***: name of dataframe column containing the units (default = 'unit')\n
            returns ->\n
            ***df***: dataframe with columns for standardized units and type of qunatity it measures 
        """

        try:

            # dict for standardization of units      
            unit_map = {
                        # fluid once variations to fl_oz 
                        'fl oz': 'fl_oz', 'fl': 'fl_oz', 'floz': 'fl_oz', 'fluid': 'fl_oz',
                        # ml varaiations to ml
                        'ml': 'ml', 'milliliter': 'ml', 'millilitre': 'ml', 'l': 'l', 'liter': 'l',
                        # ounce variations to oz
                        'oz': 'oz', 'ounce': 'oz', 'ounces': 'oz',
                        # pound variations to lb
                        'lb': 'lb', 'pound': 'lb', 'pounds': 'lb',
                        # gram variations to g
                        'g': 'g', 'gram': 'g', 'grams': 'g', 'kg': 'kg',
                        # count variations to count
                        'count': 'count', 'ct': 'count', 'each': 'count', 'unit': 'count'
                        }
            
            # replace the units  with their clean lowercase versions (if not already as such but safety)
            df['clean_units'] = df[column_name].astype(str).str.lower().str.strip()

            # map the clean unit to standardized unit from unit map
            df['standardized_unit'] = df['clean_units'].map(unit_map).fillna('ambiguous')

            # dict for getting type of quantity the unit measures
            category_map = {
                            # volume based quantities
                            'fl_oz': 'volume', 'ml': 'volume', 'l': 'volume',
                            # weight based
                            'oz': 'weight', 'g': 'weight', 'lb': 'weight', 'kg': 'weight',
                            # count based and ambiguous 
                            'count': 'discrete_count', 'ambiguous': 'ambiguous'
                            }
            
            df['unit_category'] = df['standardized_unit'].replace(category_map)

            # remove clean unit column as it just add redundancy
            df.drop('clean_units', axis = 1, inplace= True)

            return df
        
        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)
        
    def normalize_quantities(self, df: pd.DataFrame) -> pd.DataFrame:

        """
            Function to normalize the different quantitites to standard quantites(SI units) and take their natural logs to ensure uniformity in item quantity\n
            params -> \n
            ***df***: dataframe to have quantities normalized\n
            returns -> \n
            ***df*** -> Modified pandas DataFrame
        """

        try: 
            
            # np.select for quick analysis 
            conditions = [
                          # volume
                          df['standardized_unit'] == 'fl_oz', 
                          df['standardized_unit'] == 'l',
                          # weight
                          df['standardized_unit'] == 'oz',
                          df['standardized_unit'] == 'lb',
                          df['standardized_unit'] == 'kg',
                          # defaults
                          df['standardized_unit'] == 'ml',
                          df['standardized_unit'] == 'g',                          
                          df['standardized_unit'] == 'count',
                          df['standardized_unit'] == 'ambiguous'
                          ]
            
            choices = [
                       # volume
                       df['quantity'] * FLOZ_TO_ML,
                       df['quantity'] * L_TO_ML,
                       # weight
                       df['quantity'] * OZ_TO_G,
                       df['quantity'] * LB_TO_G, 
                       df['quantity'] * KG_TO_G,
                       # defaults
                       df['quantity'],
                       df['quantity'],
                       df['quantity'],
                       df['quantity']                                             
                       ]
            
            df['total_normalized_quantity'] = np.select(conditions, choices, df['quantity'])

            df['log_normalized_quantity'] = np.log1p(df['total_normalized_quantity'])

            return df

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)
        
    def find_claims(self, df: pd.DataFrame, col_name: str = 'catalog_content') -> pd.DataFrame:

        """
            Function to find claims present in catalog content such as non-GMO, Gluten Free, organically made etc and one hot 
            encode them in separate columns. \n
            params -> \n
            ***df***: Dataframe from which the extraction is to be performed on. \n
            ***col_name***: Name of the dataframe column from which the features are to be extracted. \n
            returns -> \n 
            ***df***: modified pandas DataFrame  
        """

        try:
            # create claims dict
            claim_map = {
                'is_organic': 'organic',
                'is_non_gmo': 'non-gmo|non gmo',
                'is_gluten_free': 'gluten-free|gluten free',
                'is_keto': 'keto',
                'is_vegan': 'vegan',
                'is_kosher': 'kosher'
            }

            #convert col_name content to lower case
            content_lower = df[col_name].astype(str).str.lower()
            
            # check through col_name content for any claims and set those to 1|0
            for col, keyword in claim_map.items():
                df[col] = content_lower.str.contains(keyword, regex=True, na=False).astype(np.int8)
            
            return df

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)    

    def store_processed_data(self, df: pd.DataFrame):

        """
            Function to store the given Dataframe into preprocessed file path. \n
            params ->\n 
            ***df***: Dataframe to be split\n
            returns -> None
        """

        try:
            
            logging.info(f"Pre-Processing Dataframe stored in directory {self.ingested_data_path}")

            if not os.path.exists(self.processed_file_directory):
                logging.info(f"Creating Directory {self.processed_file_directory}") 
                os.makedirs(self.processed_file_directory)

            # save in file path mentioned in config
            df.to_csv(self.preprocessed_file_path)
            
            logging.info(f"Processed Dataframe and stored in directory {self.preprocessed_file_path}")
 
        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)
        
    def plot_column_charts(self, df: pd.DataFrame): 

        """
        """

        try:

            all_columns = read_yaml(self.schema_file_path)
            
            columns = []

            columns.append(all_columns['numerical_columns'])
            columns.append(all_columns['categorical_columns'])

            with mlflow.start_run(run_name='Preprocessing stage'): 
                
                for column in columns:
                    
                    fig,ax = plt.subplots() 
                    sns.histplot(df[column], kde = True, ax = ax)
                    ax.set_title(f"Distirbution of column {column}")

                    mlflow.log_figure(fig, f"column_plots/{column}.png")


        except Exception as e:
            raise ProjectError(e, sys) 
        
    def initiate_preprocessing(self) -> PreprocessingArtifact:

        """
            Function to pre-process the DataFrame returned by the ingestion component and store the pre-processed 
            DataFrame as a csv in the pre-processed data directory.\n
            params -> None\n
            returns -> PreProcessingArtifact Dataclass Object containing preprocessed file path  
        """

        try:

            # read ingested dataframe
            df: pd.DataFrame = pd.read_csv(self.ingested_data_path)

            #apply basic preprocessing
            df['catalog_content'] = df['catalog_content'].apply(self.basic_processing)

            # find quantity and unit
            df['quantity'], df['unit'] = zip(*df['catalog_content'].apply(self.find_quantity_and_unit))

            # standardize the units
            df = self.standardize_unit(df, 'unit')

            # normalize the quantities and extract any claims
            df = self.normalize_quantities(df)
            df = self.find_claims(df, 'catalog_content')

            self.plot_column_charts(df)

            # store processed data
            self.store_processed_data(df)

            return PreprocessingArtifact(self.preprocessed_file_path)

        except Exception as e:
            logging.error(e)
            raise ProjectError(e, sys)