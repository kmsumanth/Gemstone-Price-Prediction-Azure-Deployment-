import sys
import os 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #handing feature scaling
from sklearn.impute import SimpleImputer  # handlingmissing values
from sklearn.preprocessing import OrdinalEncoder #ordinal encoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated ')
            categorical_cols=['cut', 'color', 'clarity']
            numerical_col=['carrat', 'depth', 'table', 'x', 'y', 'z']
            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VVS2','VVS1','VS2','VS1','IF']

                        #numerical pipeline
            logging.info("Pipeline initiated ")
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler()),
                ]
            )

            #categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_col),

            ('cat_pipeline',cat_pipeline,categorical_cols)
            ]
            )

            return preprocessor

            logging.info('Pipeline completed ')


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed ')
            logging.info(f'Train DataFrame Head :\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head :\n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            preprocessing_obj=self.get_data_transformation()
        except Exception as e:
            logging.info("Exception occurred in the initiate data transformation")
            raise CustomException(e,sys)
          
        