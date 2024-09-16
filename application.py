from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
     if request.method=='GET':
         res="Output:"
         return render_template('index.html',results=res)
     elif request.method=='POST':
         data = CustomData(
             gender=request.form.get('gender'),
             age=int(request.form.get('age')),
             hypertension=request.form.get('hypertension'),
             heart_disease=request.form.get('heart_disease'),
             married=request.form.get('married'),
             work_type=request.form.get('work_type'),
             residence_type=request.form.get('residence_type'),
             avg_glucose_level=float(request.form.get('avg_glucose_level')),
             bmi=float(request.form.get('bmi')),
             smoking_status=request.form.get('smoking_status')
         )

         pred_df=data.get_data_as_dataframe()
         print(pred_df)
         print("Before Prediction")

         predict_pipeline=PredictPipeline()
         print("Mid Prediction")
         results=predict_pipeline.predict(pred_df)
         print("after Prediction")
         return render_template('index.html',results=results)


if __name__=='__main__':
    app.run(host="0.0.0.0")         
