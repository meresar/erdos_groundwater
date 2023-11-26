# contains the tuned parameters for the CNN and LSTM models

# CNN parameters
##########################################################################
## RMSE: 2.3402760903147106
CNN_params_AEK201 = {'model__BATCH_SIZE': 32,
                 'model__C1_FILTERS': 64,
                 'model__C1_KERNEL': 16,
                 'model__C1_LAYER': True,
                 'model__C2_FILTERS': 64,
                 'model__C2_KERNEL': 12,
                 'model__C2_LAYER': True,
                 'model__C3_FILTERS': 64,
                 'model__C3_KERNEL': 36,
                 'model__C3_LAYER': True,
                 'model__D_MAX_LAYERS': 8,
                 'model__D_MIN_UNITS': 3,
                 'model__D_TOP_UNITS': 150,
                 'model__D_UNIT_SCALE': 0.75,
                 'model__EPOCHS': 100,
                 'model__LEARNING_RATE': 0.001,
                 'model__STOP_DELTA': 0.1}

## RMSE: 2.45326918726607
CNN_params_AFL259 = {'model__BATCH_SIZE': 32,
                 'model__C1_FILTERS': 32,
                 'model__C1_KERNEL': 8,
                 'model__C1_LAYER': True,
                 'model__C2_FILTERS': 128,
                 'model__C2_KERNEL': 24,
                 'model__C2_LAYER': True,
                 'model__C3_FILTERS': 64,
                 'model__C3_KERNEL': 18,
                 'model__C3_LAYER': True,
                 'model__D_MAX_LAYERS': 8,
                 'model__D_MIN_UNITS': 3,
                 'model__D_TOP_UNITS': 150,
                 'model__D_UNIT_SCALE': 0.75,
                 'model__EPOCHS': 100,
                 'model__LEARNING_RATE': 0.001,
                 'model__STOP_DELTA': 0.1}

## RMSE: 1.9353950671555313
CNN_params_APK309 = {'model__BATCH_SIZE': 32,
                 'model__C1_FILTERS': 16,
                 'model__C1_KERNEL': 8,
                 'model__C1_LAYER': True,
                 'model__C2_FILTERS': 32,
                 'model__C2_KERNEL': 24,
                 'model__C2_LAYER': True,
                 'model__C3_FILTERS': 64,
                 'model__C3_KERNEL': 18,
                 'model__C3_LAYER': True,
                 'model__D_MAX_LAYERS': 8,
                 'model__D_MIN_UNITS': 3,
                 'model__D_TOP_UNITS': 150,
                 'model__D_UNIT_SCALE': 0.75,
                 'model__EPOCHS': 100,
                 'model__LEARNING_RATE': 0.001,
                 'model__STOP_DELTA': 0.1} 

## RMSE: 2.416546971184388
CNN_params_APK310 = {'model__BATCH_SIZE': 32,
                 'model__C1_FILTERS': 32,
                 'model__C1_KERNEL': 8,
                 'model__C1_LAYER': True,
                 'model__C2_FILTERS': 128,
                 'model__C2_KERNEL': 24,
                 'model__C2_LAYER': True,
                 'model__C3_FILTERS': 64,
                 'model__C3_KERNEL': 18,
                 'model__C3_LAYER': True,
                 'model__D_MAX_LAYERS': 8,
                 'model__D_MIN_UNITS': 3,
                 'model__D_TOP_UNITS': 150,
                 'model__D_UNIT_SCALE': 0.75,
                 'model__EPOCHS': 100,
                 'model__LEARNING_RATE': 0.001,
                 'model__STOP_DELTA': 0.1}



#LSTM params
#############################################################################
LSTM_params_AEK201 = {'model__CHECKPOINT': False,
                        'model__D_MAX_LAYERS': 4,
                        'model__D_MIN_UNITS': 3,
                        'model__D_TOP_UNITS': 8,
                        'model__D_UNIT_SCALE': 0.5,
                        'model__EPOCHS': 30,
                        'model__LEARNING_RATE': 0.001,
                        'model__LSTM_UNITS': 64,
                        'model__NUM_FEATS': 12,
                        'model__RANDOM_STATE': 90210,
                        'model__WINDOW_SIZE': 30}


LSTM_params_AFL259 = {'model__CHECKPOINT': False,
                         'model__D_MAX_LAYERS': 4,
                         'model__D_MIN_UNITS': 3,
                         'model__D_TOP_UNITS': 8,
                         'model__D_UNIT_SCALE': 0.1,
                         'model__EPOCHS': 30,
                         'model__LEARNING_RATE': 0.0005,
                         'model__LSTM_UNITS': 64,
                         'model__NUM_FEATS': 12,
                         'model__RANDOM_STATE': 90210,
                         'model__WINDOW_SIZE': 45}

LSTM_params_APK309 = {'model__CHECKPOINT': False,
                        'model__D_MAX_LAYERS': 8,
                        'model__D_MIN_UNITS': 3,
                        'model__D_TOP_UNITS': 30,
                        'model__D_UNIT_SCALE': 0.1,
                        'model__EPOCHS': 30,
                        'model__LEARNING_RATE': 0.0005,
                        'model__LSTM_UNITS': 32,
                        'model__NUM_FEATS': 12,
                        'model__RANDOM_STATE': 90210,
                        'model__WINDOW_SIZE': 45}


LSTM_params_APK310 = {'model__CHECKPOINT': False,
                        'model__D_MAX_LAYERS': 1,
                        'model__D_MIN_UNITS': 3,
                        'model__D_TOP_UNITS': 32,
                        'model__D_UNIT_SCALE': 0.5,
                        'model__EPOCHS': 30,
                        'model__LEARNING_RATE': 0.001,
                        'model__LSTM_UNITS': 16,
                        'model__NUM_FEATS': 12,
                        'model__RANDOM_STATE': 90210,
                        'model__WINDOW_SIZE': 30}



wells = ['AEK201', 'AFL259', 'APK309', 'APK310']
CNN_well_params = [CNN_params_AEK201, 
                    CNN_params_AFL259, 
                    CNN_params_APK309, 
                    CNN_params_APK310]

LSTM_well_params = [LSTM_params_AEK201, 
                    LSTM_params_AFL259, 
                    LSTM_params_APK309, 
                    LSTM_params_APK310]