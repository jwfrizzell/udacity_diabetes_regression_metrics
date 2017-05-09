import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
	random_state=1)

reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)#Fit training set to Decision tree regression.
mean_squared_error_dtr = mse(y_test, reg1.predict(X_test))
print("Decision Tree mean absolute error: {:.2f}".format(mean_squared_error_dtr))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)#Fit training set to Linear regression model.
mean_squared_error_lr = mse(y_test,reg2.predict(X_test))

print("Linear regression mean absolute error: {:.2f}".format(mean_squared_error_lr))

results = {
 "Linear Regression": mean_squared_error_lr,
 "Decision Tree": mean_squared_error_dtr
}