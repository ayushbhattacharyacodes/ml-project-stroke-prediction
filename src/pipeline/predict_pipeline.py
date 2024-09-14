import sys,os
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
       pass

    def predict(self,features):
     try:
        model_path = os.path.join("artifacts","model.pkl")
        preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
        
        print("Before Loading")
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
       
        print("After Loading")
        data_scaled = preprocessor.transform(features)
        cols = preprocessor.get_feature_names_out()
        data = pd.DataFrame(data_scaled,columns=cols)
        columns_to_drop=[
                'cat_pipeline__gender_Male',
                'cat_pipeline__married_No',
                'cat_pipeline__work_type_children',
                'cat_pipeline__residence_type_Urban',
                'cat_pipeline__smoking_status_Unknown'
         ]
        data=data.drop(columns=columns_to_drop)
        preds = model.predict(np.array(data))
        return 'Output: You don\'t have a stroke' if preds[0] == 0.0 else 'Output:You are likely to have a stroke' 
     except Exception as e:
        raise CustomException(e,sys)


class CustomData:
   def __init__(self,
                gender:str,
                age,
                hypertension,
                heart_disease,
                married:str,
                work_type:str,
                residence_type:str,
                avg_glucose_level,
                bmi,
                smoking_status:str,
                ):
      self.gender = gender
      self.age = age
      self.hypertension =hypertension
      self.heart_disease = heart_disease
      self.married = married
      self.work_type =work_type
      self.residence_type=residence_type
      self.avg_glucose_level = avg_glucose_level
      self.bmi = bmi
      self.smoking_status=smoking_status
   def get_data_as_dataframe(self): 
    try:
       data_dict ={
          'gender':[self.gender],
          'age': [self.age],
          'hypertension':[self.hypertension],
          'heart_disease':[self.heart_disease],
          'married':[self.married],
          'work_type':[self.work_type],
          'residence_type':[self.residence_type],
          'avg_glucose_level':[self.avg_glucose_level],
          'bmi':[self.bmi],
          'smoking_status':[self.smoking_status],
       }
       df = pd.DataFrame(data_dict)
       return df
    except Exception as e:
        raise CustomException(e,sys) 

