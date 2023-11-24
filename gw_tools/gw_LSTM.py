import numpy as np
import pandas as pd

import keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from sklearn.base import BaseEstimator,TransformerMixin

## Defining a custom sklearn estimator to implement our keras model
## inside a sklearn pipeline

'''
This version is to run the model.
It assumes that the test set that is passed to the predict function
already contains the warmup period
'''
class gw_LSTM(BaseEstimator,TransformerMixin):
    def __init__(self, 
                 WINDOW_SIZE=6,
                 NUM_FEATS=12,
                 LSTM_UNITS=64,
                 D_MAX_LAYERS = 1, D_TOP_UNITS = 8, D_MIN_UNITS=2,
                 D_UNIT_SCALE = 0.5,
                 LEARNING_RATE=0.0005, LOSS=MeanSquaredError(), 
                 EPOCHS=100, 
                 RANDOM_STATE = 90210,
                 CHECKPOINT = False,
                 tmean = 0,
                 tsd = 1):
        
        self.RANDOM_STATE = RANDOM_STATE
        if self.RANDOM_STATE is not None:
            keras.utils.set_random_seed(RANDOM_STATE)

        self.WINDOW_SIZE = WINDOW_SIZE
        self.NUM_FEATS = NUM_FEATS
        
        self.LSTM_UNITS = LSTM_UNITS
        
        self.D_MAX_LAYERS = D_MAX_LAYERS
        self.D_TOP_UNITS = D_TOP_UNITS
        self.D_MIN_UNITS = D_MIN_UNITS
        self.D_UNIT_SCALE = D_UNIT_SCALE

        self.LEARNING_RATE = LEARNING_RATE
        self.LOSS = LOSS

        self.EPOCHS = EPOCHS

        self.CHECKPOINT = CHECKPOINT

        self.model = None

        self.tmean = tmean
        self.tsd = tsd
        

    def reshape_data(self, scaled_np, y=None):
        # getting dumb errors so we're just gonna double check that we're
        # starting with a numpy array
        scaled_np = np.array(scaled_np)
        X = []
        for i in range(len(scaled_np)-self.WINDOW_SIZE):
            row = [a for a in scaled_np[i:i+self.WINDOW_SIZE]]
            X.append(row)
        #print(np.array(X)[0,0], np.array(X).shape)
        return np.array(X)
        
    def make_model(self):
        ## Initialize a sequential model object
        model = Sequential()

        ## Add the input layer
        model.add(InputLayer((self.WINDOW_SIZE, self.NUM_FEATS)))

        ## Add LSTM layer
        model.add(LSTM(self.LSTM_UNITS))
        

        ## Flattening layer
        model.add(Flatten())
        
        ## Add Dense Layers
        d_layers = 0
        d_units = self.D_TOP_UNITS
        while ((d_layers < self.D_MAX_LAYERS) and (d_units > self.D_MIN_UNITS)):
            model.add(Dense(d_units, activation='relu'))
            d_layers += 1
            d_units = int(d_units*self.D_UNIT_SCALE)

        ## Output Layer
        model.add(Dense(1, activation='linear'))
        
        
        ## Compile the model
        model.compile(loss=self.LOSS, 
                    optimizer=Adam(learning_rate=self.LEARNING_RATE), 
                    metrics=[RootMeanSquaredError()])
        
        ## Return the model
        self.model = model


    def fit(self,X_train,y_train):
        self.make_model()

        ## reshape the data to go into the fit
        X_window = self.reshape_data(X_train)

        if self.CHECKPOINT == False:
            self.model.fit(X_window, y_train[self.WINDOW_SIZE:], 
                            epochs=self.EPOCHS)
        else:
            self.model.fit(X_window, y_train[self.WINDOW_SIZE:], 
                            epochs=self.EPOCHS, 
                            callbacks=[self.CHECKPOINT])
            # in order to implement this, we need to plug a val set into the model fitting, and I haven't worked out the details for how to do that, so just setting checkpoint as false all the time now I think

        
    def transform(self,X,y=None):
        self.model.transform(X,y)

    def predict(self,X_test,y_test=None):
        ## X_test already contains the warmup set in this case
        ## so it has length of TEST_SIZE+WINDOW_SIZE
        X_df = pd.DataFrame(X_test)
        #display(X_df)

        preds=[]

        for i in range(X_test.shape[0]-self.WINDOW_SIZE):
            # reshape a row at a time
            row = self.reshape_data(np.array(X_df[i:i+self.WINDOW_SIZE+1]))
            #print(row)
            
            # make prediction and store it
            pred = self.model.predict(row).flatten()[0]
            preds.append(pred)
            #print(preds)
            
            #insert prediction into the correct place for the next loop
            #print('before')
            X_df[0][self.WINDOW_SIZE+i] = (pred - self.tmean) / self.tsd
            #print('value set to', X_df[0][self.WINDOW_SIZE+i])

        return np.array(preds)

    def score(self,X,y=None):
        pred = self.predict(X)
        score = np.mean((pred - y)**2)
        return -score
