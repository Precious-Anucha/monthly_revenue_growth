import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('Monthly Revenue Prediction')

# Load Data
df1 = pd.read_csv('Branch_01.csv')
df2 = pd.read_csv('Branch_02.csv')
df3 = pd.read_csv('Branch_03.csv')

df = pd.concat([df1, df2, df3])
df['Date'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df = df.sort_values(by=['Branch_ID', 'Month'])

# removing duplicates 
df.drop_duplicates(inplace=True)

# showing the first 5 rows of the initial dataframe
st.write(df.head())


# Feature Engineering
df['Prev_Total_Deposits'] = df.groupby('Branch_ID')['Total_Deposits'].shift(1)
df['Prev_Loan_Approvals'] = df.groupby('Branch_ID')['Loan_Approvals'].shift(1)
df['Prev_Revenue_Growth'] = df.groupby('Branch_ID')['Revenue_Growth'].shift(1)
df.dropna(inplace=True)

le = LabelEncoder()
df['Branch_Name'] = le.fit_transform(df['Branch_Name'])

# Splitting the Data
features = ['Branch_ID', 'Branch_Name', 'Month','Total_Deposits', 'Prev_Total_Deposits', 'Prev_Loan_Approvals', 'Prev_Revenue_Growth', 'Customer_Satisfaction_Score']
target = 'Revenue_Growth'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Training DataSet After preprocessing")
st.write(X_train.head())


# Train Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluation
y_pred = lr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write("### Model Evaluation")
st.write(f"MAE: {mae}")
st.write(f"RMSE: {rmse}")
st.write(f"R² Score: {r2}")

# User Input for Prediction
st.write("### Predict Revenue Growth")
branch_id = st.number_input("Branch ID", min_value=int(df['Branch_ID'].min()), max_value=int(df['Branch_ID'].max()), step=1)
branch_name = st.number_input("Branch Name (Encoded)", min_value=int(df['Branch_Name'].min()), max_value=int(df['Branch_Name'].max()), step=1)
current_total_deposits = st.number_input("Total Deposits", min_value=0.0, step=1000.0)
prev_total_deposits = st.number_input("Previous Total Deposits", min_value=0.0, step=1000.0)
loan_approvals = st.number_input("Previous Loan Approvals", min_value=0, step=1)
prev_revenue_growth = st.number_input("Previous Revenue Growth", min_value=0.0, step=0.1)
customer_satisfaction = st.slider("Customer Satisfaction Score", min_value=0, max_value=10, step=1)

if st.button("Predict Revenue Growth"):
    input_data = pd.DataFrame({
        'Branch_ID': [branch_id],
        'Branch_Name': [branch_name],
        'Total_Deposits': [current_total_deposits],
        'Prev_Total_Deposits': [prev_total_deposits],
        'Prev_Loan_Approvals': [loan_approvals],
        'Prev_Revenue_Growth': [prev_revenue_growth],
        'Customer_Satisfaction_Score': [customer_satisfaction]
    })
    prediction = lr_model.predict(input_data)
    st.write(f"Predicted Revenue Growth: {prediction[0]:.2f}")

