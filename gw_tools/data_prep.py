import numpy as np
import pandas as pd
import pickle


## Paths to the pickles
AEK201_path = '../data/pickled_data/AEK201_short.pkl'
AFL259_path = '../data/pickled_data/AFL259_short.pkl'
APK309_path = '../data/pickled_data/APK309_short.pkl'
APK310_path = '../data/pickled_data/APK310_short.pkl'

WELLS = {'AEK201':AEK201_path,
         'AFL259':AFL259_path,
         'APK309':APK309_path,
         'APK310':APK310_path}

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
    if WELL in WELLS:
        with open(WELLS[WELL], 'rb') as f:
            df = pickle.load(f)
    else:
        with open(WELL, 'rb') as f:
            df = pickle.load(f)
    return df

DEFAULT_FEATURES = ['date', 'avg_well_depth', 'gage_ht', 'discharge_cfs',
                        'prcp','temp_avg', 'hum_avg', 'hPa_avg', 'wind_avg',
                        'gust_avg', 'prcp_lag_45D']

def select_features(df, features=DEFAULT_FEATURES):
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
    return  df[features].copy()

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


