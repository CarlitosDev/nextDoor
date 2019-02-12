import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn import preprocessing
from datetime import datetime
import time
import uuid
from random import random
import itertools

class nextDoorForecaster:
    '''
        Forecasting engine class based on kNN+feature learning

        Carlos Aguilar, 08.02.19

        autopep8 --in-place --aggressive recommender.py

        Updates:
         - 

        Full documentation:


    '''

    # Private attributes
    __queryStart = []
    __queryEnd = []
    __queryElapsed = []
    __createdAt = datetime.now()
    __uuid = str(uuid.uuid1())

    # Set the normaliser public
    min_max_scaler = preprocessing.MinMaxScaler(copy=False)
    min_vals = []
    max_vals = []

    # training split
    training_split = [];

    # Algo
    __X_train = []
    __Y_train = []
    featWeight = []
    rnorm = []
    kNeighbours = 15
    # if a weight is inf, it will be represented
    # as the max finite weight times maxWeigthScale
    maxWeightScale = 15


    # Constructor
    def __init__(self, training_split=0.25):
        self.training_split = training_split

    def __scale_X_train(self, x):
        '''
            Normalise training data between 0 and 1
        '''
        # Normalise input data between 0 and 1
        self.min_max_scaler.fit_transform(x)
        self.min_vals = self.min_max_scaler.data_min_
        self.max_vals = self.min_max_scaler.data_max_
        return x

    def __scale_X_test(self, x_test):
        '''
            Normalise test data with the values found 
            during training
            
        '''
        self.min_max_scaler.transform(x_test)
        return x_test

    def __solve_nnls_training(self, X, Y):

        self.__queryStart = time.time()
        numRecords = X.shape[0]
        testSize   = round(numRecords*self.training_split)

        idx_test = np.random.choice(numRecords, testSize, replace=False)

        M = []
        e = []
        currentPromo    = np.zeros(numRecords, dtype=bool)
        remainingPromos = np.ones(numRecords, dtype=bool)

        for k in range(0, testSize):
            idx = idx_test[k]
            currentPromo[idx]    = True
            remainingPromos[idx] = False
            M.append(np.power(X[remainingPromos] - X[currentPromo], 2))
            e.append(np.power(Y[remainingPromos] - Y[currentPromo], 2))
            currentPromo[idx]    = False
            remainingPromos[idx] = True

        M = np.concatenate(M, axis=0).copy()
        e = np.concatenate(e, axis=0).copy()

        featWeight, rnorm  = nnls(M, e)
        self.featWeight = featWeight
        self.rnorm = rnorm
        print(f'NNLS size {M.shape[0]}x{M.shape[1]}', end='')
        self.__setElapsed__()

    def __setElapsed__(self):
        self.__queryEnd     = time.time()
        self.__queryElapsed = self.__queryEnd - self.__queryStart
        print('...done in {:.2f} sec!'.format(self.__queryElapsed))

    @staticmethod
    def normalise_vector(vector_a, max_val):
        '''
            Normalise VECTOR_A from 0 to MAX_VAL

            Some recommendations have got the same score, 
            so let's check out and return them as MAX_VAL
            
        '''
        minVal   = vector_a.min()
        maxVal   = vector_a.max()
        rangeVal = maxVal - minVal
        if rangeVal == 0.0:
            vector_a[:] = max_val
        else:
            vector_a = max_val*(vector_a-minVal) / (maxVal-minVal)
        return vector_a

    @staticmethod
    def get_basic_stats(vector_a):
        '''
            Info VECTOR_A
            
        '''
        d = {'max': vector_a.max(), 
        'min': vector_a.min(),
        'mean': vector_a.mean()}

        return d


    # Public methods
    def train(self, X_train, y_train):
        self.__scale_X_train(X_train)
        self.__solve_nnls_training(X_train, y_train)
        self.__X_train = X_train
        self.__Y_train = y_train
        # set the max number of neighbours available
        self.kNeighbours = np.min([len(y_train), self.kNeighbours])

    # Public methods
    def cv_neighbours(self, X_val, y_val):
        self.__scale_X_test(X_val)
        y_hat     = np.zeros([y_val.shape[0], self.kNeighbours-1])
        frc_error = np.zeros([y_val.shape[0], self.kNeighbours-1])

        for idx, cpromo in enumerate(X_val, 0):
            currentWeightsSorted, Y_trainSorted,_,_ = self.calculateWeights(cpromo)
            for k in range(1, self.kNeighbours):
                normalisedWeights = currentWeightsSorted[0:k]/sum(currentWeightsSorted[0:k])
                currentFrc = normalisedWeights.dot(Y_trainSorted[0:k])
                y_hat[idx][k-1] = currentFrc
                frc_error[idx][k-1] = currentFrc - y_val[idx]

        frc_error_mu  = np.mean(frc_error, axis=0)
        frc_abs_error = np.abs(frc_error_mu)

        self.kNeighbours = 1 + np.argmin(frc_abs_error)

        return y_hat[:, self.kNeighbours-1], frc_error[:, self.kNeighbours-1]

    def calculateWeights(self, x_test):
        M_test         = np.power(self.__X_train-x_test, 2)
        currentWeights = np.power(M_test.dot(self.featWeight), -1/2)
        idxInf         = np.isinf(currentWeights)
        maxWeight      = np.max(np.append(currentWeights[~idxInf], 1.0))
        # Adjust to finite
        currentWeights[idxInf] = self.maxWeightScale*maxWeight
        # order descending
        idxSorted = np.argsort(currentWeights)[::-1]
        
        currentWeightsSorted = currentWeights[idxSorted]
        Y_trainSorted = self.__Y_train[idxSorted]
        return currentWeightsSorted, Y_trainSorted, idxSorted, currentWeights

    def predict(self, X_test):
        self.__scale_X_test(X_test)
        y_hat = []
        for cpromo in X_test:
            # Adjust to finite
            currentWeightsSorted, Y_trainSorted,_,_ = self.calculateWeights(cpromo)
            
            normalisedWeights = currentWeightsSorted[0:self.kNeighbours]/sum(currentWeightsSorted[0:self.kNeighbours])
            currentFrc = normalisedWeights.dot(Y_trainSorted[0:self.kNeighbours])
            
            print(f'Forecast {currentFrc:.2f}')
            y_hat.append(currentFrc)

        return np.array(y_hat)


    @staticmethod
    def get_frc_errors(y, y_hat):
        '''
            Get forecast residuals as e_t = \hat{y} - y
                so e_t > 0 overforecast
                so e_t < 0 underforecast
                e_t = 0 : the dream
            
        '''
        e_t =  y_hat-y
        abs_e_t = np.abs(e_t)
        MAE = abs_e_t.mean()
        MSE = np.power(e_t, 2).mean()
        meanError = e_t.mean()
        MAPE = 100*(abs_e_t/np.abs(y)).mean()
        d = {'MAE': MAE,
        'MSE': MSE, 
        'meanError': meanError,
        'MAPE': MAPE, 
        'residuals': e_t}
        return d