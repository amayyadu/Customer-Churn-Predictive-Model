import pandas as pd
import numpy as np
import os
csv_path = os.path.join(os.path.dirname(__file__), 'churnguard_data.csv') if '__file__' in locals() else 'churnguard_data.csv'
df = pd.read_csv(csv_path)

print(df.head())
print(df.shape)

print(df.info())
print('No. of NA Values :\n',df.isna().sum())
print(f'No. of duplicates : {df.duplicated().sum()}')
print(df['Churn'].value_counts())
print(df['Contract'].unique())

df=df.drop('customerID',axis=1)
df=df.drop_duplicates()
df['gender']=df['gender'].str.strip()
df['PaymentMethod']=df['PaymentMethod'].str.strip()
for col in ['Churn', 'PhoneService', 'PaperlessBilling']:
    df[col] = df[col].str.strip().str.title()
    print(df[col].value_counts())

df['Contract']=df['Contract'].str.strip().str.title()

mapping = {
    'Monthly': 'Month-To-Month',
    'Month To Month': 'Month-To-Month',
    '1 Year': 'One Year',
    '2 Year': 'Two Year'
}
df['Contract']=df['Contract'].replace(mapping)
df['Contract'].value_counts()

df['InternetService']=df['InternetService'].str.strip().str.title()
df['InternetService']=df['InternetService'].replace({'Fibre Optic':'Fiber Optic','Fiberoptic':'Fiber Optic'})
df['InternetService'].value_counts().sum()

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df=df[df['tenure']>0]
df=df[(df['MonthlyCharges']>=10) & (df['MonthlyCharges']<=200)]
df['MonthlyCharges']=df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure']=df['tenure'].fillna(df['tenure'].median())

print(df.shape)
df.isna().sum()

df['Churn']=(df['Churn']=='Yes').astype(int)
column=['gender', 'PhoneService', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df=pd.get_dummies(df,columns=column, drop_first=True)
X=df.drop('Churn',axis=1)
y=df['Churn']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f'Accuracy : {accuracy_score(y_test,y_pred)}')
print(classification_report(y_test,y_pred,target_names=['Stay', 'Churn']))

scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X)
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_full_scaled, y)
user_tenure = int(input("Enter tenure (months): "))
user_monthly = float(input("Enter Monthly Charges: "))
user_total = float(input("Enter Total Charges: "))
user_senior = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
user_contract_code = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))
contract_map = {0: 'Month-To-Month', 1: 'One Year', 2: 'Two Year'}
user_contract_text = contract_map.get(user_contract_code, 'Month-To-Month')
user_raw_dict = {
    'SeniorCitizen': [user_senior],
    'tenure': [user_tenure],
    'MonthlyCharges': [user_monthly],
    'TotalCharges': [user_total],
    'Contract': [user_contract_text],
    'gender': ['Male'],
    'PhoneService': ['No'],
    'InternetService': ['Dsl'],
    'PaperlessBilling': ['No'],
    'PaymentMethod': ['Mailed Check']
}
user_df = pd.DataFrame(user_raw_dict)
user_encoded = pd.get_dummies(user_df, columns=column, drop_first=True)
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0).astype(int)
user_scaled = scaler.transform(user_encoded)
prediction = model_full.predict(user_scaled)

if prediction[0] == 1:
    print("\nPrediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")
