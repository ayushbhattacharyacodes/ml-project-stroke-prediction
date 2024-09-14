import sys,os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import BorderlineSMOTE

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]
            target_column=["stroke"]
            categorical_columns=["gender","married","work_type","residence_type","smoking_status"]

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("one_hot_encoder",OneHotEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def remove_cols(self,X,cols):
        try:
            self.X=X.drop(columns=cols)
            return self.X
        except Exception as e:
            raise CustomException(e,sys)
        
    def apply_bsmote(self,x,y):
        try:
            smote=BorderlineSMOTE(sampling_strategy='minority')
            x_resampled,y_resampled = smote.fit_resample(x,y)
            return x_resampled,y_resampled
        
        except Exception as e:
            raise CustomException(e,sys)    


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
             
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="stroke"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Transformation Completed Successfully')
            features = preprocessing_obj.get_feature_names_out()
            transformed_train_df = pd.DataFrame(input_feature_train_arr,columns=features)
            transformed_test_df = pd.DataFrame(input_feature_test_arr,columns=features)

            logging.info('Dropping a few categorical columns after transformation')
            columns_to_drop=[
                'cat_pipeline__gender_Male',
                'cat_pipeline__married_No',
                'cat_pipeline__work_type_children',
                'cat_pipeline__residence_type_Urban',
                'cat_pipeline__smoking_status_Unknown'
            ]
            transformed_train_df=self.remove_cols(transformed_train_df,columns_to_drop)
            transformed_test_df=self.remove_cols(transformed_test_df,columns_to_drop)

            logging.info("Preprocessing completed successfully")

            logging.info("Obtaining BorderlineSMOTE method and applying it on training dataframe")
            input_feature_train_array,target_feature_train_array = self.apply_bsmote(np.array(transformed_train_df),np.array(target_feature_train_df)) 
            input_feature_test_array,target_feature_test_array = self.apply_bsmote(np.array(transformed_test_df),np.array(target_feature_test_df))  
            
            logging.info("Application of BorderlineSMOTE completed")
            
            
            train_arr = np.c_[input_feature_train_array, target_feature_train_array]
            test_arr = np.c_[input_feature_test_array, target_feature_test_array]
            
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)    
