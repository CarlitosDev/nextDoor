import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
import time
import uuid
from random import random
from joblib import Parallel, delayed

class nextDoorForecasterV2:
    '''
        Forecasting engine class based on kNN+feature learning
        Version 2


        Carlos Aguilar
          v1 - 08.02.19
          v2 - 11.09.19

        Updates:
          11.09.19 - Pass lambda regularisation effectively
          13.09.19 - Calculate correlation 
          21.09.19 - Fix bug in comparing element-wise matrices
          22.09.19 - Silence the 0-division (unlikely in real data scenarios)




        Full documentation in "Forecasting Promotional Sales Within the Neighbourhood"
        https://ieeexplore.ieee.org/document/8727882

    '''

    # Private attributes
    __queryStart = []
    __queryEnd = []
    __queryElapsed = []
    __uuid = str(uuid.uuid1())
    
    createdAt = datetime.now()

    # Set the normaliser public
    min_max_scaler = preprocessing.MinMaxScaler(copy=False)
    min_vals = []
    max_vals = []
    scale    = []

    # training split
    training_split = [];

    # Algo
    __X_train = []
    __Y_train = []
    featWeight = []
    num_nnls_batches = []
    features_batches = []

    rnorm = []
    kNeighbours = 15
    # if a weight is inf, it will be represented
    # as the max finite weight times maxWeigthScale
    maxWeightScale = 15
    # Validation
    val_set_error = np.Inf
    val_set_forecast = np.Inf


    # Constructor
    def __init__(self, training_split=0.25, num_nnls_batches = 0):
        self.training_split = training_split
        self.num_nnls_batches = num_nnls_batches

    def __scale_X_train(self, x):
        '''
            Normalise training data between 0 and 1
        '''
        # Normalise input data between 0 and 1
        #self.min_max_scaler.fit_transform(x)
        #?????
        self.min_max_scaler = self.min_max_scaler.fit(x)
        self.min_max_scaler.transform(x)
        # Copy the parameters as I am not able to get the scaler fit between calls (?!)
        self.min_vals = self.min_max_scaler.data_min_
        self.max_vals = self.min_max_scaler.data_max_
        self.min_max_scale = self.min_max_scaler.scale_
        return x

    def __scale_X_test(self, x_test):
        '''
            Normalise test data with the values found 
            during training
            
        '''
        try:
            self.min_max_scaler.transform(x_test)
        except NotFittedError as e:
            # Notify if for any reason it's not fitted
            print('min_max_scaler - not fitted')
        return x_test

    def __solve_nnls_training(self, X, Y, _lambda):

        numRecords  = X.shape[0]
        numFeatures = X.shape[1]

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

        # If regularisation
        if _lambda > 0.0:
            M_prime = _lambda*np.eye(numFeatures)
            e_prime = np.zeros(numFeatures)
            M = np.append(M, M_prime, axis = 0)
            e = np.append(e, e_prime, axis = 0)


        featWeight, rnorm  = nnls(M, e)
        self.featWeight = featWeight
        self.rnorm = rnorm
        #print(f'NNLS size {M.shape[0]}x{M.shape[1]}', end='')

        

    def __solve_nnls_training_batches(self, X, Y, _lambda):
    
        all_numRecords  = X.shape[0]
        idx = np.arange(all_numRecords)
        np.random.shuffle(idx)
        idx_step = np.split(idx, 3)

        all_featWeight = []
        all_rnorm = []

        for this_idx in idx_step:

            this_X = X[this_idx]
            this_y = Y[this_idx]

            numRecords  = this_X.shape[0]
            numFeatures = this_X.shape[1]

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
                M.append(np.power(this_X[remainingPromos] - this_X[currentPromo], 2))
                e.append(np.power(this_y[remainingPromos] - this_y[currentPromo], 2))
                currentPromo[idx]    = False
                remainingPromos[idx] = True

            M = np.concatenate(M, axis=0).copy()
            e = np.concatenate(e, axis=0).copy()

            # If regularisation
            if _lambda > 0.0:
                M_prime = _lambda*np.eye(numFeatures)
                e_prime = np.zeros(numFeatures)
                M = np.append(M, M_prime, axis = 0)
                e = np.append(e, e_prime, axis = 0)

            featWeight, rnorm  = nnls(M, e)
            all_featWeight.append(featWeight)
            all_rnorm.append(rnorm)

        self.features_batches = all_featWeight
        self.featWeight = np.mean(all_featWeight, axis=0)
        self.rnorm = np.mean(all_rnorm, axis=0)

    # Dummie getters to see what's going on
    def get_X_train(self):
        return self.__X_train
    def get_Y_train(self):
        return self.__Y_train

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


    # Train the Nearest Neighbour
    def train_NN(self, X_train, y_train, _lambda=0.0):
        self.__scale_X_train(X_train)
        if self.num_nnls_batches==0:
            self.__solve_nnls_training(X_train, y_train, _lambda)
        else:
            self.__solve_nnls_training_batches(X_train, y_train, _lambda)
        self.__X_train = X_train
        self.__Y_train = y_train
        # set the max number of neighbours available
        self.kNeighbours = np.min([len(y_train), self.kNeighbours])

    # Validate the number of neighbours
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
        self.val_set_error = frc_error[:, self.kNeighbours-1]
        self.val_set_forecast = y_hat[:, self.kNeighbours-1]

    def calculateWeights(self, x_test):
        M_test         = np.power(self.__X_train-x_test, 2)
        # ignore the warnings as the Inf values are removed from the weights
        with np.errstate(divide='ignore'):
            currentWeights = np.power(M_test.dot(self.featWeight), -1/2)
        # With this one that checks for zeroes
        #product_Mw     = M_test.dot(self.featWeight)
        #idx_non_zero = product_Mw != 0.0
        #currentWeights = np.Inf*np.ones_like(product_Mw)
        #currentWeights[idx_non_zero] = np.power(product_Mw[idx_non_zero], -1/2)
        
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

    def train_and_validate(self, X_train: 'numpy ND array', y_train: 'numpy 1D array', X_val=[], y_val=[], _lambda=0.0):
      '''
        Train and validate using Next Door Neighbours

          If the validation sets are empty, they will be randomly drawn from the training one

      '''
      if (not isinstance(X_val, np.ndarray)) and (not isinstance(y_val, np.ndarray)):
        # Overwrite the arrays
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


      self.train_NN(X_train.copy(), y_train.copy(), _lambda=_lambda)
      self.cv_neighbours(X_val.copy(), y_val.copy())


    @staticmethod
    def train_only(X_train, y_train, X_val, y_val, _lambda=0.0, _training_split=0.5, _num_nnls_batches=0):
        '''
          Return a trained ND forecaster
        '''

        forecaster = nextDoorForecasterV2(training_split=_training_split, \
            num_nnls_batches=_num_nnls_batches)
        
        forecaster.train_and_validate(X_train.copy(), y_train.copy(), 
          X_val=X_val.copy(), y_val=y_val.copy(), _lambda=_lambda)

        return forecaster


    @staticmethod
    def one_go(X_train, y_train, X_val, y_val, X_test, _lambda=0.0, _training_split=0.5):

        forecaster = nextDoorForecasterV2(training_split=_training_split)
        
        forecaster.train_and_validate(X_train.copy(), y_train.copy(), 
        X_val=X_val.copy(), y_val=y_val.copy(), _lambda=_lambda)
        
        predictions = forecaster.predict(X_test)
        return [predictions,forecaster.kNeighbours,forecaster.featWeight]

    @staticmethod
    def train(X, Y, X_val, y_val, num_forecasters=100, _lambda=0.0, _training_split=0.5, _num_nnls_batches=0):
        queryStart = time.time()
        r = Parallel(n_jobs=-1)(delayed(nextDoorForecasterV2.train_only)(X,Y,X_val,y_val,_lambda, _training_split, _num_nnls_batches) for i in range(num_forecasters))
        queryElapsed = time.time() - queryStart
        print(f'...training {num_forecasters} forecasters done in {queryElapsed:.2f} sec!')
        return r

    @staticmethod
    def fit(X, Y, X_val, y_val, X_test, num_forecasters=100, _lambda=0.0, _training_split=0.5):
        queryStart = time.time()
        r = Parallel(n_jobs=-1)(delayed(nextDoorForecasterV2.one_go)(X,Y,X_val,y_val,X_test,_lambda, _training_split) for i in range(num_forecasters))
        queryElapsed = time.time() - queryStart
        print(f'...prediction with {num_forecasters} forecasters done in {queryElapsed:.2f} sec!')
        t_predictions = []
        kNeighbours   = []
        featWeight    = []
        for item in r:
            t_predictions.append(item[0])
            kNeighbours.append(item[1])
            featWeight.append(item[2])
        # aggregated results for predictions
        all_predictions = np.array(t_predictions)
        predictions     = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)
        predictions_max = np.max(all_predictions, axis=0)
        predictions_min = np.min(all_predictions, axis=0)
        # aggregate neighbours and features

        num_neighbours = np.array(kNeighbours).mean()
        features       = np.array(featWeight).mean(axis=0)

        return {'predictions': predictions, 
        'predictions_std': predictions_std, 
        'predictions_min': predictions_min, 
        'predictions_max': predictions_max,
        'num_neighbours': num_neighbours,
        'features': features}

    @staticmethod
    def getLinearScaler(inputData: 'np.array vector', y_max, y_min):
        #[fcn_scaler, fcn_denormaliser, m_slope, b_intercept]
        '''

        x = [inputData.min(), inputData.max()]
        A = np.vstack([x, np.ones(len(x))]).T
        '''

        x = [inputData.min(), inputData.max()]
        A = np.vstack([x, np.ones(len(x))]).T
        y = [y_min, y_max]
        m_slope, b_intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        fcn_scaler       = lambda x: m_slope*x + b_intercept
        fcn_denormaliser = lambda x_trans: (x_trans-b_intercept)/m_slope;

        return {'fcn_scaler': fcn_scaler, 
        'fcn_denormaliser': fcn_denormaliser, 
        'm_slope': m_slope, 
        'b_intercept': b_intercept}

    @staticmethod
    def predict_with_ensemble(ensemble, X_input: 'must be np.array (not a DF)'):
        '''
        Predict using a ensemble of NDN's
        '''
        t_predictions = []
        for frc in ensemble:
            t_predictions.append(frc.predict(X_input.copy()))
        all_predictions = np.array(t_predictions)
        predictions     = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)
        return {'predictions': predictions, 
            'predictions_std': predictions_std}