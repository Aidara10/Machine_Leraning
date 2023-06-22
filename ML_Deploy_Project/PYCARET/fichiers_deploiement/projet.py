# 1. Library imports
from joblib import load
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = load_model('ChurnModel')


# Define predict function
@app.post('/predict')
def predict(state, account_length, area_code, international_plan,voice_mail_plan,number_vmail_messages,total_day_minutes,total_day_calls,total_day_charge,total_eve_minutes, total_eve_calls,total_eve_charge,total_night_minutes, total_night_calls, total_night_charge,total_intl_minutes,	total_intl_calls, total_intl_charge, customer_service_calls):
    data = pd.DataFrame([[state, account_length, area_code,international_plan,voice_mail_plan,number_vmail_messages,total_day_minutes,total_day_calls,total_day_charge,total_eve_minutes,	total_eve_calls,total_eve_charge,total_night_minutes, total_night_calls, total_night_charge,total_intl_minutes,	total_intl_calls, total_intl_charge, customer_service_calls]])
    data.columns = ['state', 'account_length', 'area_code','international_plan','voice_mail_plan','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge','total_eve_minutes',	'total_eve_calls','total_eve_charge','total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes','total_intl_calls', 'total_intl_charge', 'customer_service_calls']

    predictions = predict_model(model, data=data)
    return {'prediction': int(predictions['Label'][0])}



if __name__ == '__projet__':
    uvicorn.run(app, host='127.0.0.1', port=8000)