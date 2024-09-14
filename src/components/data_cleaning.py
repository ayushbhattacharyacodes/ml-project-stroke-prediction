import os,sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exceptions import CustomException

@dataclass
class DataCleaningConfig:
    train_data_path:str = os.path.join('artifacts','train.csv') 
    test_data_path:str = os.path.join('artifacts','test.csv')


class DataCleaning:
    def __init__(self):
        self.data_cleaning_config = DataCleaningConfig()

    def remove_duplicates(self,X):
        try:
                self.X = X.drop_duplicates(keep='first')
                return self.X
        except Exception as e:
            raise CustomException(e,sys)

    def replace_values(self,X,col,val):
        try:
                return X[col].replace(val,X[col].mode()[0])
        except Exception as e:
            raise CustomException(e,sys)
        
    def drop_columns(self,X,columns):
        try:
                self.X=X.drop(columns=columns,axis=1)
                return  self.X
        except Exception as e:
            raise CustomException(e,sys)
        
    def replace_nan(self,X,col):
        try:
                self.X[col]=X[col].replace(np.nan,X[col].median())
                return self.X[col] 
        except Exception as e:
            raise CustomException(e,sys)
    
    def rename_columns(self,X,rename_map):
         try:
              return X.rename(columns=rename_map)
         except Exception as e:
              raise CustomException(e,sys)    
         

    def initiate_data_cleaning(self,raw_data_path):
         logging.info('Entered Data Cleaning Configuration')

         try:
              data = pd.read_csv(raw_data_path)
              logging.info('Read the dataset as a dataframe')
              
              logging.info('Create a copy of the original dataframe')
              df = data.copy(deep=True)
              logging.info('Copy created successfully')


              columns_to_drop = ["id"]
              rename_mapping = {'ever_married':'married','Residence_type':'residence_type'}
              replace_tuple = ('gender','bmi')
             
              logging.info('Removing the duplicated values')
              df = self.remove_duplicates(df)
              logging.info('Duplicates removed successfully')

              logging.info('Replacing the value for categorical column')
              df[replace_tuple[0]] = self.replace_values(df,replace_tuple[0],'Other')
              logging.info('Values for categorical column replaced successfully')
              
              logging.info('Dropping the unnecessary columns')
              df = self.drop_columns(df,columns_to_drop)
              logging.info('Dropped the columns successfully')
              
              logging.info('Replacing the value for numerical column')
              df[replace_tuple[1]] = self.replace_nan(df,replace_tuple[1]) 
              logging.info('Values for numerical column replaced successfully')

              logging.info('Renaming certain columns')
              df = self.rename_columns(df,rename_mapping)
              logging.info('Renamed columns successfully')

              logging.info('Splitting the data into training and testing dataset')
              train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)
              logging.info('Splittong completed successfully')

              logging.info('Saving the splitted data')
              train_data.to_csv(self.data_cleaning_config.train_data_path,index=False,header=True)
              test_data.to_csv(self.data_cleaning_config.test_data_path,index=False,header=True)
              
              logging.info('Data cleaning completed')
              
              return(
                   self.data_cleaning_config.train_data_path,
                   self.data_cleaning_config.test_data_path
              )
         
         except Exception as e:
            raise CustomException(e,sys)  