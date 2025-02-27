import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('Monthly Revenue Prediction')

# Load Data
df1 = pd.read_csv('Branch_01.csv')
df2 = pd.read_csv('Branch_02.csv')
df3 = pd.read_csv('Branch_03.csv')
df4 = pd.read_csv('Branch_04.csv')

# merging the various dataframes into one dataframe
df = pd.concat([df1, df2, df3, df4])

st.title('Data Before preprocessing')
df = df.sort_values(by=['Branch_ID', 'Date'])
st.write(df.head())

df['Date'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
# df = df.sort_values(by=['Branch_ID', 'Date'])

# Feature Engineering - Adding Lag & Rolling Features
df['Prev_Total_Deposits'] = df.groupby('Branch_ID')['Total_Deposits'].shift(1)
df['Prev_Loan_Approvals'] = df.groupby('Branch_ID')['Loan_Approvals'].shift(1)
#df['Prev_Revenue_Growth'] = df.groupby('Branch_ID')['Revenue_Growth'].shift(1)

# Feature Engineering - difference between the current month and previous month
df['change_in_deposit'] = df['Total_Deposits'] - df['Prev_Total_Deposits']
df['change_in_loan_approvals'] = df['Loan_Approvals'] - df['Prev_Loan_Approvals']
#df['change_in_revenue_growth'] = df['Revenue_Growth'] - df['Prev_Revenue_Growth']

# 3-Month Rolling Features
df['Rolling_Deposits_3M'] = df.groupby('Branch_ID')['Total_Deposits'].rolling(3).mean().reset_index(level=0, drop=True)
df['Rolling_Loan_Approvals_3M'] = df.groupby('Branch_ID')['Loan_Approvals'].rolling(3).mean().reset_index(level=0, drop=True)

# Interaction Features
df['Deposits_x_Satisfaction'] = df['Total_Deposits'] * df['Customer_Satisfaction_Score']
df['Loan_x_Satisfaction'] = df['Loan_Approvals'] * df['Customer_Satisfaction_Score']

df.dropna(inplace=True)

# Encode Branch_Name before splitting
le = LabelEncoder()
df['Branch_Name'] = le.fit_transform(df['Branch_Name'])

# Outlier Removal (IQR Method)
Q1 = df['Revenue_Growth'].quantile(0.25)
Q3 = df['Revenue_Growth'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Revenue_Growth'] >= Q1 - 1.5 * IQR) & (df['Revenue_Growth'] <= Q3 + 1.5 * IQR)]

# Define features and target
features = ['Branch_ID', 'Branch_Name', 'Month', 'Total_Deposits', 'Prev_Total_Deposits', 
            'Loan_Approvals', 'Prev_Loan_Approvals', 'Prev_Revenue_Growth', 'Rolling_Deposits_3M',
            'Rolling_Loan_Approvals_3M', 'Deposits_x_Satisfaction', 'Loan_x_Satisfaction', 'Customer_Satisfaction_Score', 'change_in_deposit', 'change_in_loan_approvals', ']
target = 'Revenue_Growth'

X = df[features]
y = df[target]

# Time-Based Train-Test Split (instead of random)
split_date = df['Date'].quantile(0.8)  # Use 80% for training
X_train, X_test = X[df['Date'] <= split_date], X[df['Date'] > split_date]
y_train, y_test = y[df['Date'] <= split_date], y[df['Date'] > split_date]

st.write("Training Data After Preprocessing")
st.write(X_train.head())

# Standard Scaling
scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

# Train Model - Using Random Forest
rf_model = LinearRegression()
rf_model.fit(X_train_ss, y_train)

# Evaluation
y_pred = rf_model.predict(X_test_ss)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write("### Model Evaluation (After Improvements)")
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# User Input for Prediction
st.write("### Predict Revenue Growth")
branch_id = st.number_input("Branch ID", min_value=int(df['Branch_ID'].min()), max_value=int(df['Branch_ID'].max()), step=1)
branch_name = st.selectbox("Branch Name", df['Branch_Name'].unique())
current_total_deposits = st.number_input("Total Deposits", min_value=0.0, step=1000.0)
prev_total_deposits = st.number_input("Previous Total Deposits", min_value=0.0, step=1000.0)
loan_approvals = st.number_input("Loan Approvals", min_value=0, step=1)
prev_revenue_growth = st.number_input("Previous Revenue Growth", min_value=0.0, step=0.1)
customer_satisfaction = st.slider("Customer Satisfaction Score", min_value=0, max_value=10, step=1)

if st.button("Predict Revenue Growth"):
    input_data = pd.DataFrame({
        'Branch_ID': [branch_id],
        'Branch_Name': [branch_name],  # Encoded
        'Month': [df['Month'].max()],
        'Total_Deposits': [current_total_deposits],
        'Prev_Total_Deposits': [prev_total_deposits],
        'Loan_Approvals': [loan_approvals],
        'Prev_Loan_Approvals': [loan_approvals],  
        'Prev_Revenue_Growth': [prev_revenue_growth],
        'Rolling_Deposits_3M': [current_total_deposits],  # Assume rolling avg ≈ latest value
        'Rolling_Loan_Approvals_3M': [loan_approvals],  # Assume rolling avg ≈ latest value
        'Deposits_x_Satisfaction': [current_total_deposits * customer_satisfaction],
        'Loan_x_Satisfaction': [loan_approvals * customer_satisfaction],
        'Customer_Satisfaction_Score': [customer_satisfaction]
    })

    # Scale input
    input_data_ss = scaler.transform(input_data)
    prediction = rf_model.predict
