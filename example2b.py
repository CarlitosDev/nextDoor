import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from nextDoorForecaster import nextDoorForecaster
from math import sqrt
from joblib import Parallel, delayed

dataRoot  = './data'

trainingSet = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/PythonDevs/trainingSet3.xlsx'
testSet = '/Users/carlosAguilar/Google Drive/order/Machine Learning Part/PythonDevs/testSet3.xlsx'


df_training = pd.read_excel(trainingSet)
df_test = pd.read_excel(testSet)



# input vars
numericalVarNames = ['total_stores', 
  'total_store_revenue', 
  'offer_price', 
  'for_price', 
  'current_price', 
  'discount',
  'numericalOSD']

inputVars   = numericalVarNames
responseVar = 'wk1_sales_all_stores'

df_training.shape[0]


# get datasets
X_train = df_training.iloc[0:550][inputVars].values
y_train = df_training.iloc[0:550][responseVar].values

# validate
X_val = df_training.iloc[550::][inputVars].values
y_val = df_training.iloc[550::][responseVar].values

# test
X_test = df_test[inputVars].values
y_test = df_test[responseVar].values





# Run the forecaster in parallel
num_frcs = 50
#lambda_value = 100000
lambda_value = 0.0
_training_split = 0.75
d_predictions = nextDoorForecaster.fit(X_train,y_train,X_val,y_val,X_test,num_frcs, lambda_value, _training_split)
y_hat = d_predictions['predictions']
print(y_hat)
errors = nextDoorForecaster.get_frc_errors(y_test, y_hat)
print(f'{num_frcs} forecasters with MSE {errors["MSE"]:.2f} and MAPE {errors["MAPE"]:.2f} and mError {errors["meanError"]:.2f}')

nV = nextDoorForecaster.normalise_vector(d_predictions['features'], 100)
var_importance = pd.DataFrame(nV, index=inputVars)
var_importance





##

forecaster = nextDoorForecaster(training_split=0.5)
forecaster.train(X_train.copy(),y_train.copy())
forecaster.train(X_train.copy(),y_train.copy(), 0.15)
_,_= forecaster.cv_neighbours(X_val.copy(), y_val.copy())

predictions = forecaster.predict(X_test)

#
forecaster.__solve_nnls_training(X_train.copy(),y_train.copy(), _lambda=0.0)