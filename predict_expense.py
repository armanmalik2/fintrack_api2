#CSV file se data Extraction in variable x and y
import pandas as pd

df = pd.read_csv('FinTrack_Expense_Salary_Data1.csv')

x = df[['Year','Month','Salary']]
y = df['Total_Expenses']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_scaled,y)
run = model.predict(scaler.transform([[2140,10,200000]]))
print(run)


import joblib

joblib.dump((model), 'expenses.joblib')
