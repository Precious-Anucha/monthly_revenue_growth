import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('🎈 App Name')

df1 = pd.read_csv('Branch_01.csv')
df2 = pd.read_csv('Branch_02.csv')
df3 = pd.read_csv('Branch_03.csv')

df = pd.concat([df1, df2, df3])

df['Date'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month

# df = df.sort_values(by=['Branch_ID', 'Month'])
df = df.sort_values(by=['Branch_ID', 'Month'])

print(df.info())

pd.set_option('display.max_columns', None)


# Feature Engineering


# feature Selection
#df.drop('Relationship_Manager_Name', inplace = True)

#print(df.corr())

df['Prev_Total_Deposits'] = df.groupby('Branch_ID')['Total_Deposits'].shift(1)
df['Prev_Loan_Approvals'] = df.groupby('Branch_ID')['Loan_Approvals'].shift(1)
df['Prev_Revenue_Growth'] = df.groupby('Branch_ID')['Revenue_Growth'].shift(1)
df.dropna(inplace=True)


le = LabelEncoder()
df['Branch_Name'] = le.fit_transform(df['Branch_Name'])




# Splitting the data
features = ['Branch_ID', 'Branch_Name', 'Relationship_Manager_ID', 'Prev_Total_Deposits',
            'Prev_Loan_Approvals', 'Prev_Revenue_Growth', 'Customer_Satisfaction_Score']
target = 'Revenue_Growth'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df.head())

# instantiamte the linearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.write(mae, rmse, r2)


print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

df['Month'] = pd.to_datetime(df['Month'])
next_month = df[df['Month'] == df['Month'].max()].copy()


next_month['Month'] = next_month['Month'] + pd.DateOffset(months=1)

# Use previous month’s data as features
next_month['Prev_Total_Deposits'] = next_month['Total_Deposits']
next_month['Prev_Loan_Approvals'] = next_month['Loan_Approvals']
next_month['Prev_Revenue_Growth'] = next_month['Revenue_Growth']

next_month_predictions = lr_model.predict(next_month[features])
next_month['Predicted_Revenue_Growth'] = next_month_predictions

print(next_month[['Branch_ID', 'Month', 'Predicted_Revenue_Growth']])


st.write('Hello world!')

# st.markdown(f"**{}**")
#     st.write()
#     st.write()
    # fig, ax = plt.subplots(figsize=(4, 4))
    # wedges, _ = ax.pie([prediction_percentage, 100 - prediction_percentage], startangle=90, colors=[color, "lightgrey"], wedgeprops=dict(width=0.3))
    # plt.title(prediction_label)
    # st.pyplot(fig)
