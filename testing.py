import joblib
import pandas as pd

model = joblib.load("loan_pipeline.pkl")

new_data = pd.DataFrame([{
    "Age": 40,
    "Income": 60,
    "LoanAmount": 2000000,
    "CreditScore": 650,
    "MonthsEmployed": 24,
    "NumCreditLines": 3,
    "InterestRate": 10.5,
    "LoanTerm": 36,
    "DTIRatio": 0.3,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "No",
    "HasDependents": "Yes",
    "LoanPurpose": "Auto",
    "HasCoSigner": "No"
}])

pd_value = model.predict_proba(new_data)[0][1]
score = int(900 - (pd_value * 600))

print("PD:", pd_value)
print("Risk Score:", score)