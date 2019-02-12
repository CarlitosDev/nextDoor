import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from nextDoorForecaster import nextDoorForecaster

dataRoot  = '/Users/carlos.aguilar/Google Drive/order/Machine Learning Part/PythonDevs'
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

# Instanciate the forecaster
forecaster = nextDoorForecaster(training_split=0.5)

# Solve NNLS
X = df.iloc[0:300][inputVars].values
Y = df.iloc[0:300][responseVar].values

# mind the values in X will be modified
forecaster.train(X,Y)
#forecaster.train(df[inputVars].values,df[responseVar].values)
print(forecaster.normalise_vector(forecaster.featWeight, 100))

df.describe()

# validate
X_val = df.iloc[300:400][inputVars].values
y_val = df.iloc[300:400][responseVar].values
y_hat, frc_error = forecaster.cv_neighbours(X_val, y_val)
print(f'CV neighbours {forecaster.kNeighbours}')

print(forecaster.get_basic_stats(frc_error))

# test
X_test = df.iloc[400::][inputVars].values
y_test = df.iloc[400::][responseVar].values
y_hat = forecaster.predict(X_test)

errors = forecaster.get_frc_errors(y_test, y_hat)
errors['MSE']
errors['meanError']
errors['MAPE']



x_test = X_test[0]
currentWeightsSorted, Y_trainSorted,_,_ =forecaster.calculateWeights(x_test)



0









