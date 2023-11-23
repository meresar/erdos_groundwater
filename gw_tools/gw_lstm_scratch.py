import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Flatten
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from keras.optimizers.legacy import Adam
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

## Defining a custom sklearn estimator to implement our keras model
## inside a sklearn pipeline
class gw_lstm(BaseEstimator,TransformerMixin):
    def __init__(self, WINDOW_SIZE=6):
        self.model = None
        self.WINDOW_SIZE = WINDOW_SIZE
        self.NUM_FEATS = 11
        self.EPOCHS = 20
        self.BATCH_SIZE = 32

    def reshape_data(self, X, y=None):
        X_list = []
        y_list = []
        for i in range(X.shape[0]-self.WINDOW_SIZE):
            row = [a for a in X[i:i+self.WINDOW_SIZE]]
            X_list.append(row)
            label = y[i+self.WINDOW_SIZE]
            y_list.append(label)
        return np.array(X_list), np.array(y_list)
        
    def make_model(self, X, y=None):
        self.NUM_FEATS = X.shape[1]
        
        # Reshape the data for the LSTM
        X, y = self.reshape_data(X, y)

        ## Construct the model
        model = Sequential()
        model.add(Input((self.WINDOW_SIZE, self.NUM_FEATS)))
        model.add(LSTM(64))
        model.add(Flatten())
        model.add(Dense(8, 'relu'))
        model.add(Dense(1, 'linear'))
        
        ## Compile the model
        model.compile(loss=MeanSquaredError(), 
                      optimizer=Adam(learning_rate=0.001), 
                      metrics=[RootMeanSquaredError()])
        
        ## Return the model
        self.model = model

    def fit(self,X,y=None):
        #callback = EarlyStopping(monitor='loss', 
        #                         patience=4, 
        #                         min_delta=self.STOP_DELTA)
        
        # Reshape the data for the LSTM
        X, y = self.reshape_data(X, y)
        
        checkpoint = ModelCheckpoint('mereLSTM/', save_best_only=True)
        
        self.make_model(X,y)
        
        self.model.fit(X,y, 
                       epochs=self.EPOCHS, 
                       batch_size=self.BATCH_SIZE,
                       callbacks=[checkpoint])
        
    def transform(self,X,y=None):
        #self.make_model(X,y)
        self.model.transform(X,y)

    def predict(self,X,y=None):
        return self.model.predict(X,y)

    def score(self,X,y=None):
        pred = self.model.predict(X)
        score = np.mean((pred - y)**2)
        return -score
