import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 1.	In Folder Q1, there is a dataset in which we aim to estimate the house price using two features: the number of bedrooms and the basement area.

# Use Multiple Linear Regression for this task. Display the coefficients of the model and calculate the MAE (Mean Absolute Error) and MSE (Mean Squared Error). Search about RMSE (Root Mean Squared Error) and explain the trade-offs between these metrics. Finally report RMSE score of your model.
# Perform this task using both LinearRegression and SGDRegressor.
# Additionally, study the MAPE (Mean Absolute Percentage Error) metric using this link, and apply it to evaluate your model.

# DATA
dataset = pd.read_csv('./house_price.csv')
X = dataset [['size', 'bedroom']]
y = dataset['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# MODEL
# Using Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print("Linear Regression Coefficients: \n")
print(coeff_df)
print("Intercept: ", regressor.intercept_)


# Using SGD Regressor
sgdRegressor = SGDRegressor(max_iter=1000)
sgdRegressor.fit(X_train, y_train)
coeff_df_sgd = pd.DataFrame(sgdRegressor.coef_, X.columns, columns=['Coefficient'])
print("\nSGD Regressor Coefficients: \n")
print(coeff_df_sgd)
print("Intercept: ", sgdRegressor.intercept_)


# RESULTS AND EVALUATION
y_pred = regressor.predict(X_test)
print("\nLinear Regression:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred))

y_pred_sgd = sgdRegressor.predict(X_test)
print("\nSGD Regressor:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_sgd))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_sgd))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_sgd)))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred_sgd))



# RMSE is calculated as the square root of the Mean Squared Error (MSE). It penalizes larger errors more heavily due to the squaring process and is expressed in the same units as the predicted variable, making it easy to interpret.

# MAE simply calculates the average of the absolute differences between the predicted and actual values. It treats all errors equally, which makes it more robust to outliers. The downside is that it doesn’t reflect how far off large errors may be.

# MSE also penalizes larger errors more heavily, but its values are in squared units, which can make them harder to interpret directly.
