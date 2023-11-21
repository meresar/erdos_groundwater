import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

## Defining a custom sklearn estimator to implement our keras model
## inside a sklearn pipeline
class gw_cnn(BaseEstimator,TransformerMixin):
    def __init__(self, 
                 WINDOW_SIZE=6,
                 LSTM_UNITS=64,
                 D_MAX_LAYERS = 1, D_TOP_UNITS = 8, D_MIN_UNITS=2,
                 D_UNIT_SCALE = 0.5,
                 LEARNING_RATE=0.0005, LOSS=MeanSquaredError(), 
                 EPOCHS=100, BATCH_SIZE=64, STOP_DELTA=.05,
                 RANDOM_STATE = 90210):
        
        self.RANDOM_STATE = RANDOM_STATE
        if self.RANDOM_STATE is not None:
            keras.utils.set_random_seed(RANDOM_STATE)
        
        self.LSTM_UNITS = LSTM_UNITS
        
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

        

    def reshape_data(self, scaled_np, y=None):
        X = []
        y = []
        for i in range(len(scaled_np)-self.WINDOW_SIZE):
            row = [a for a in scaled_np[i:i+self.WINDOW_SIZE]]
            X.append(row)
            label = scaled_np[i+self.WINDOW_SIZE][0]
            y.append(label)
        return np.array(X), np.array(y)
        
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

    def fit(self,X,y=None):
        callback = EarlyStopping(monitor='loss', patience=4, 
                                    min_delta=self.STOP_DELTA)
        self.make_model(X,y)
        self.model.fit(X,y, 
                        epochs=self.EPOCHS,
                        callbacks=[callback])
        
    def transform(self,X,y=None):
        self.model.transform(X,y)

    def predict(self,X,y=None):
        #I think my loop is gonna go here
        return self.model.predict(X,y)

    def score(self,X,y=None):
        pred = self.model.predict(X)
        score = np.mean((pred - y)**2)
        return -score
