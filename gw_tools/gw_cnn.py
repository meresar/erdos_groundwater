import keras
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.losses import MeanSquaredError
from keras.optimizers.legacy import Adam
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

## Defining a custom sklearn estimator to implement our keras model
## inside a sklearn pipeline
class gw_cnn(BaseEstimator,TransformerMixin):
    def __init__(self, 
                 C1_LAYER=True, C1_FILTERS=32, C1_KERNEL=4,
                 C2_LAYER=True, C2_FILTERS=64, C2_KERNEL=9,
                 C3_LAYER=True, C3_FILTERS=128, C3_KERNEL=9,
                 D_MAX_LAYERS = 4, D_TOP_UNITS = 80, D_MIN_UNITS=15,
                 D_UNIT_SCALE = 0.5,
                 LEARNING_RATE=0.0005, LOSS=MeanSquaredError(), 
                 EPOCHS=100, BATCH_SIZE=64, STOP_DELTA=.05,
                 RANDOM_STATE = 90210):
        
        self.RANDOM_STATE = RANDOM_STATE
        if self.RANDOM_STATE is not None:
            keras.utils.set_random_seed(RANDOM_STATE)
        
        self.C1_LAYER = C1_LAYER
        self.C1_FILTERS = C1_FILTERS
        self.C1_KERNEL = C1_KERNEL
        
        self.C2_LAYER = C2_LAYER
        self.C2_FILTERS = C2_FILTERS
        self.C2_KERNEL = C2_KERNEL
        
        self.C3_LAYER = C3_LAYER
        self.C3_FILTERS = C3_FILTERS
        self.C3_KERNEL = C3_KERNEL
        
        self.D_MAX_LAYERS = D_MAX_LAYERS
        self.D_TOP_UNITS = D_TOP_UNITS
        self.D_MIN_UNITS = D_MIN_UNITS
        self.D_UNIT_SCALE = D_UNIT_SCALE

        self.LEARNING_RATE = LEARNING_RATE
        self.LOSS = LOSS

        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
       
        self.STOP_DELTA = STOP_DELTA

        self.RANDOM_STATE = RANDOM_STATE

        self.model = None

        

    def reshape_data(self, X, y=None):
        return X.reshape((X.shape[0], X.shape[1], 1))
        
    def make_model(self, X, y=None):
        # Reshape the data for the CNN
        X = self.reshape_data(X)

        ## Initialize a sequential model object
        model = Sequential()
        ## Add the input layer
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        
        ## Add Convolutional layers
        if self.C1_LAYER:
            model.add(Conv1D(filters=self.C1_FILTERS, 
                                kernel_size=self.C1_KERNEL, 
                                strides=1, padding='same', activation='relu'))
        if self.C2_LAYER:
            model.add(Conv1D(filters=self.C2_FILTERS, 
                                kernel_size=self.C1_KERNEL, 
                                strides=1, padding='same', activation='relu'))
        if self.C3_LAYER:
            model.add(Conv1D(filters=self.C3_FILTERS, 
                                kernel_size=self.C1_KERNEL, 
                                strides=1, padding='same', activation='relu'))
        
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
        model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE), 
                                        loss=self.LOSS)
        
        ## Return the model
        self.model = model

    def fit(self,X,y=None):
        callback = EarlyStopping(monitor='loss', patience=4, 
                                    min_delta=self.STOP_DELTA)
        self.make_model(X,y)
        self.model.fit(X,y, 
                        epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                        callbacks=[callback])
        
    def transform(self,X,y=None):
        #self.make_model(X,y)
        self.model.transform(X,y)

    def predict(self,X,y=None):
        return self.model.predict(X,y)

    def score(self,X,y=None):
        pred = self.model.predict(X)
        score = np.mean((pred - y)**2)
        return -score
