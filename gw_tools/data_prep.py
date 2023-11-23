import datetime
import numpy as np
import pandas as pd
import pickle


## Paths to the pickles
AEK201_path = '../data/pickled_data/AEK201_short.pkl'
AFL259_path = '../data/pickled_data/AFL259_short.pkl'
APK309_path = '../data/pickled_data/APK309_short.pkl'
APK310_path = '../data/pickled_data/APK310_short.pkl'
FEATS_path = '../data/pickled_data/all_feature_data.pkl'

DKEYS = {'AEK201':AEK201_path,
         'AFL259':AFL259_path,
         'APK309':APK309_path,
         'APK310':APK310_path,
         'FEATS':FEATS_path}

## Start Dates
well_start_dates = {'AEK201':datetime.datetime(2005,8,22),
                    'AFL259':datetime.datetime(2005,8,21),
                    'APK309':datetime.datetime(2006,6,21),
                    'APK310':datetime.datetime(2006,6,21)}

## End Dates
well_end_dates = {'AEK201':datetime.datetime(2017,9,28),
                  'AFL259':datetime.datetime(2017,9,28),
                  'APK309':datetime.datetime(2017,9,28),
                  'APK310':datetime.datetime(2017,5,4)}


def load_data(WELL=None):
    '''Load data for a well from a pickle.

    Parameters
    ----------
    WELL : string
        A key from the WELLS dictionary, or file path to a .pkl file.
    
    Returns
    -------
        pandas.DataFrame
        The dataframe that was loaded from the pickle.
    '''
    if WELL in DKEYS:
        with open(DKEYS[WELL], 'rb') as f:
            df = pickle.load(f)
    else:
        with open(WELL, 'rb') as f:
            df = pickle.load(f)
    return df



def select_features(df, features=None, no_target=False):
    ''' Select the desired subset of features
        
        Parameters
        ----------
        df : pandas.DataFrame
            Assumed to be of the format that all of our wells are save in

        features : list(string) (Default: DEFAULT_FEATURES)
            A list of the features to be kept

        Returns
        -------
        pandas.DataFrame
            Returns a copy of df that only includes the selected features
    '''
    FEATS_AND_TARGET = ['date', 'avg_well_depth', 'gage_ht', 'discharge_cfs',
                        'prcp','temp_avg', 'hum_avg', 'hPa_avg', 'wind_avg',
                        'gust_avg', 'prcp_lag_45D']

    FEATS_ONLY = ['date', 'gage_ht', 'discharge_cfs',
                  'prcp','temp_avg', 'hum_avg', 'hPa_avg', 'wind_avg',
                  'gust_avg', 'prcp_lag_45D']

    if features is not None:
        return  df[features].copy()
    elif no_target:
        return  df[FEATS_ONLY].copy()
    else:
        return df[FEATS_AND_TARGET].copy()

def add_toy_signal(df):
    ''' Add time of year signal columns to the dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            Assumed to have a 'date' column populated with datetime objects 
    
        Returns
        -------
        pandas.DataFrame
            A copy of df with `year_sin` and `year_cos` columns added
    '''
    
    df = df.copy()

    ## Pop off date_time information
    date_time = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S').copy()

    ## Collect each date as a UNIX epoch (seconds since Jan 1, 1970)
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    
    ## Define seconds in a day, and a year
    day = 24*60*60
    year = (365.2425)*day
    
    ## Add sine and cosine signals to mark time of year
    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df.copy()

def prep_data_for_training(df, n=365):
    ''' Set aside holdout data
        
        Parameters
        ----------
        n : int (Default: 365)
            The number of days to include in the holdout set

        Returns
        -------
        This function returns the following as np.arrays():
        - X_train
            The training feature data (excludes 'date' and 'avg_well_depth')
        - X_holdout
            The holdout feature data (excludes 'date' and 'avg_well_depth')
        - y_train
            The training target data ('avg_well_depth' only)
        - y_holdout
            The holdout target data ('avg_well_depth' only)
        - dt_train
            The dates corresponding to the training data
        - dt_holdout
            The dates corresponding to the holdout data

    '''

    dates = pd.to_datetime(df.pop('date'), format='%d.%m.%Y %H:%M:%S').copy()
    target = df.pop('avg_well_depth').copy()
    features = df.copy()

    X_train = features[:-n].copy().values
    X_holdout = features[-n:].copy().values
    y_train = target[:-n].copy().values
    y_holdout = target[-n:].copy().values
    dt_train = dates[:-n].copy().values
    dt_holdout = dates[-n:].copy().values

    return X_train, X_holdout, y_train, y_holdout, dt_train, dt_holdout

def LSTM_data_prep(df, TEST_SIZE=365, WINDOW_SIZE):
    '''
    The LSTM model requires the target to be included in the inputs
    so I don't pop it off like in the prep_data_for_training above

    It also needs the training mean and standard deviation to scale
    the recursive predictions

    Parameters
    ----------
    TEST_SIZE : int (Default: 365)
        The number of days to include in the holdout set
    WINDOW_SIZE : int
        The size of the window to be considered in the model
        (may vary by well)

    Returns
    -------
    This function returns the following:
    - X_train : data frame
        The training feature data (INCLUDING avg_well_depth, date as index)
    - X_test : data frame
        The holdout feature data (INCLUDING avg_well_depth, date as index)
        includes a warmup set for the model
    - well_tr_mean : float
        the mean of the avg_well_depth on the training set
    - well_tr_std : float
        the standard deviation of the avg_well_depth on the training set   
    '''
    X_train = df[:-TEST_SIZE].copy().set_index('date')
    X_test = df[-TEST_SIZE-WINDOW_SIZE:].copy().set_index('date')

    well_tr_mean = np.mean(X_train.avg_well_depth.values)
    well_tr_std = np.std(X_train.avg_well_depth.values)

    return X_train, X_test, well_tr_mean, well_tr_std

def LSTM_future(feats, end, feats_end, end_date, WINDOW_SIZE):
    '''
    The LSTM model requires the target to be included in the inputs
    and a warmup set

    Parameters
    ----------
    feats : a data frame
        contains the features (not including the well depth) for the dates beyond where we have well data, includes the date
    end : list
        contains (at least) the last WINDOW_SIZE values for the well data
        could be predictions or actual data
    feats_end : dataframe
        contains (at least) the last WINDOW_SIZE values for the feature data
        (*including* the well depth AND the date)
        in practice, this will be the X_test as returned by the LSTM_data_prep
    end_date : datetime
        the last date in the well depth data (as returned by get_end_date)
    WINDOW_SIZE : int
        The size of the window to be considered in the model
        (may vary by well)

    Returns
    -------
    This function returns the following:
    - to_predict : dataframe
        contains the data necessary for the LSTM model to run, including a 
        warmup window
    '''
    feats_future = feats.loc[features.date > end_date].copy()
    to_predict = pd.concat([feats_end[-WINDOW_SIZE:],feats_future])
    well_vals = end[-WINDOW_SIZE:]

    for val,i in zip(well_vals,range(len(well_vals))):
        to_predict.avg_well_depth.iloc[i] = val

    return to_predict

def get_end_date(well):
    return well_end_dates[well]

def get_start_date(well):
    return well_start_dates[well]