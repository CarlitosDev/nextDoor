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
dataFile  = os.path.join(dataRoot, 'fakeArray.mat')
matData   = sio.loadmat(dataFile)

data     = matData['fakeArray']
varNames = [f[0] for f in matData['varNames'][0]]
df   = pd.DataFrame(data, columns=varNames)

numRecords = df.shape[0]


# input vars
inputVars   = ['num1', 'num2', 'confoundingVar', 
'c1_0', 'c1_1', 'c2_0', 'c2_1']
responseVar = 'responseVar'


# get datasets
X = df.iloc[0:300][inputVars].values
Y = df.iloc[0:300][responseVar].values

# validate
X_val = df.iloc[300:400][inputVars].values
y_val = df.iloc[300:400][responseVar].values

# test
X_test = df.iloc[400::][inputVars].values
y_test = df.iloc[400::][responseVar].values


# Run the forecaster in parallel
num_frcs = 40
d_predictions = nextDoorForecaster.fit(X,Y,X_val,y_val,X_test,num_frcs)
y_hat = d_predictions['predictions']
errors = nextDoorForecaster.get_frc_errors(y_test, y_hat)
print(f'{num_frcs} forecasters with MSE {errors["MSE"]:.2f} and MAPE {errors["MAPE"]:.2f} and mError {errors["meanError"]:.2f}')

nV = nextDoorForecaster.normalise_vector(d_predictions['features'], 100)
var_importance = pd.DataFrame(nV, index=inputVars)
var_importance
