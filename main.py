import streamlit as st
import pandas as pd
from predict import main

def get_user_input():
    st.title("Loan Approval Prediction Form")

    # loan_id = st.text_input("Loan ID", value="LP2023")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married?", options=["Yes", "No"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1, value=0)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", options=["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", options=[1, 0])
    property_area = st.selectbox("Property Area", options=["Urban", "Rural", "Semiurban"])

    if st.button("Submit"):
        new_data = pd.DataFrame({
            # 'Loan_ID': [loan_id],
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        st.write("Here is the input data you provided:")
        st.write(new_data)

        
        return new_data
    return None

# Run the app
if __name__ == "__main__":
    new_data = get_user_input()
    if new_data is not None:
        mapped_data=main(new_data)  
        if mapped_data[0] == "Approved":
            color = "green"
        else:
            color = "red"

        st.markdown(
            f"<div style='background-color:{color};padding:10px;border-radius:5px;'>"
            f"<h3 style='color:white;text-align:center;'>{mapped_data}</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
 

        # # Perform prediction or other actions
        # st.write("Processing the input for prediction...\n")
        # st.write(f"Loan Approval Prediction: {mapped_data}")
