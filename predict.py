import pickle
from src.model import load_data
from sklearn.preprocessing import LabelEncoder
from src.evaluate import prediction

import pandas as pd
import pickle
import streamlit as st

label_encoder=LabelEncoder()
def load_model(path='model/model.pkl'):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def get_data():
    loan_id = input("Enter Loan ID: ")
    gender = input("Enter Gender (Male/Female): ")
    married = input("Married? (Yes/No): ")
    dependents = int(input("Number of Dependents: "))
    education = input("Education (Graduate/Not Graduate): ")
    self_employed = input("Self Employed? (Yes/No): ")
    applicant_income = int(input("Applicant Income: "))
    coapplicant_income = int(input("Coapplicant Income: "))
    loan_amount = int(input("Loan Amount: "))
    loan_amount_term = int(input("Loan Amount Term (in days): "))
    credit_history = int(input("Credit History (1 for Yes, 0 for No): "))
    property_area = input("Property Area (Urban/Rural/Semiurban): ")
    new_data = pd.DataFrame({
        'Loan_ID':[loan_id],
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term ],
        'Credit_History': [credit_history],
        'Property_Area': [property_area ]
    })
    print(f"columns:{new_data.columns}")
    return new_data

def preprocess_data(df):
    df_columns=df.select_dtypes(include=['object']).columns


    for col in  df_columns:
        # print(f"col:{col}")
        df[col]=label_encoder.fit_transform(df[col])
    print(f"df:{df}")
    # value=prediction(df)
    return df

def main(df):
    label_mapping = {
        0: 'Not Approved',
        1: 'Approved'
    }
    
    model=load_model('model/model.pkl')
    # df=get_data()

    df=preprocess_data(df)
    print(df)
    value=prediction(model,df)
    print(value)
    

    mapped_predictions = [label_mapping[value[0]]]
    print(mapped_predictions)

    return mapped_predictions