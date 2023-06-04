import pandas as pd
import math
from SL.LinearRegress import OLSLinearRegression as OLS
from SL.LinearRegress import GDLinearRegression as GD
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_science/assets_dsc/Salary_Data.csv')
# df = pd.read_csv('https://raw.githubusercontent.com/pvannyamelia/dsc_kuliah/main/data_science/assets_dsc/real_estate.csv')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train,  Y_test = tts(X, Y, test_size=0.25, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

model = GD(X_train, Y_train)
Y_pred = model.predict(X_test)

model.report()
rmse = mse(Y_test, Y_pred, squared=False)
print(rmse)
# Cara lain buat cari RMSE
# rmse2 = math.sqrt(mse(Y_test, Y_pred))
# print(rmse2)

Y_pred2 = []
for y in Y_pred:
    Y_pred2.append(y)
df_compare = pd.DataFrame({'prediction' : Y_pred2, 'actual' : Y_test})
df_compare.reset_index(drop=True, inplace=True)

# print(df_compare)

plt.figure(figsize=[25,8])
plt.plot(df_compare.index,df_compare['prediction'], label='Predictied')
plt.plot(df_compare.index, df_compare['actual'], label='Actual')
plt.title('Salary Prediction Using Gradient Descent Linear Regression')
plt.xlabel('Data Index')
plt.ylabel('Salary')
plt.legend()
plt.show()