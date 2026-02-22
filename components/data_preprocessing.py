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
from sklearn.model_selection import train_test_split

# conversion constants for a single measuring unit of each type 
FLOZ_TO_ML = 29.5735
L_TO_ML = 1000
OZ_TO_G = 28.3495
LB_TO_G = 453.592
KG_TO_G = 1000


class PreProcessingComponent():

    def __init__(self, ingestion_artifact: IngestionArtifact, preprocessing_config: PreprocessingConfig) -> None:
        
        try:
            
            logging.info("----- Initializing Data Preprocessing Component -----")

            # initialize the components and read the contents of ingestion artifact dataclass
            self.ingestion_artifact = ingestion_artifact
            self.ingested_data_path = self.ingestion_artifact.ingested_data_dir

            self.config = preprocessing_config
            self.train_file_path = self.config.training_file_path
            self.test_file_path = self.config.testing_file_path

            # initialize set of stopwords to prevent re initiaslization for every string instance
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.add('\n')

        except ProjectError as e:

            raise(e)
        
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

        except ProjectError as e:
            raise(e) 

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

            quantity = np.float32(s[value_idx+8: unit_idx])

            unit = s[unit_idx + 7: ].strip()

            return (quantity, unit)
        
        except ProjectError as e:
            return (e, 'ambiguous')
        
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
                        'fl_oz': 'fl_oz', 'fl': 'fl_oz', 'floz': 'fl_oz', 'fluid': 'fl_oz',
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
            df.drop('clean_units', inplace= True)

            return df
        
        except ProjectError as e:
            raise(e)
        
    def normalize_quantities(self, df: pd.DataFrame) -> pd.DataFrame:

        """
            Function to normalize the different quantitites to standard quantites(SI units) to ensure uniformity in item quantity\n
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

        except ProjectError as e:
            raise(e)
        
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

        except ProjectError as e:
            raise(e)    

    def split_data(self, df: pd.DataFrame):

        """
            Function to split the given Dataframe into train and test subsets as per the config split ratio
            And store the train and test files in the train/test file paths respectively\n
            params ->\n 
            ***df***: Dataframe to be split\n
            returns -> None
        """

        try:
            
            logging.info(f"Splitting Dataframe stored in directory {self.ingested_data_path} in ratio {self.config.split_ratio}")

            # split as per split ratio
            train, test = train_test_split(df, test_size = self.config.split_ratio, random_state=42)

            train_path = self.config.training_file_path
            test_path = self.config.testing_file_path

            if not os.path.exists(train_path):
                logging.info(f"Creating Directory {train_path}") 
                os.makedirs(train_path)

            if not os.path.exists(test_path):
                logging.info(f"Creating Directory {test_path}") 
                os.makedirs(test_path)

            # save in file path mentioned in config
            train.to_csv(train_path)
            test.to_csv(test_path)
            
            logging.info(f"Saved Dataframe split in train and test files into directories {train_path} and {test_path}")
 
        except ProjectError as e:
            raise(e)