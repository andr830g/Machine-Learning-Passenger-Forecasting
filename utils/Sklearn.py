import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from utils.tools import *
from utils.metrics import Time
from IPython.display import clear_output
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


"""
Training methods
"""

def trainModel(model, X_train, y_train, scalar, exog_scalar, horizon, lags, differentiation, useExog, useLags):
    # difference
    if differentiation == 1:
        y_train_diff = difference(y_train)
    else:
        y_train_diff = y_train.copy(deep=True)

    # add exogenous or empty dataframe
    if useExog == True:
        X_train_temp = X_train.copy(deep=True)
    else:
        X_train_temp = pd.DataFrame(index=y_train.index)

    # add lagged values
    if useLags:
        for lag in lags:
            X_train_temp[f'lag_{lag}'] = y_train_diff.shift(periods=lag).fillna(0)

    # fit model
    X_train_lagdiff_transformed = pd.DataFrame(exog_scalar.fit_transform(X_train_temp), columns=X_train_temp.columns)
    y_train_transformed = scalar.fit_transform(y_train_diff.to_numpy().reshape(-1, 1))
    model.fit(X_train_lagdiff_transformed, y_train_transformed)
    y_train_pred_transformed = model.predict(X_train_lagdiff_transformed).reshape(-1, 1)
    y_train_pred_diff = pd.Series(scalar.inverse_transform(y_train_pred_transformed).squeeze(1))

    # inverse difference
    if differentiation == 1:
        y_train_pred = inverseDifferencing(y_train_pred_diff, y_train, horizon)
    else:
        y_train_pred = y_train_pred_diff.copy(deep=True)

    return model, y_train_pred, scalar, exog_scalar


"""
Fixed window methods
"""

def fixedWindowWithLags(X_train, y_train, X_val, y_val, model, horizon, differentiation, lags, scalar, exog_scalar, useExog, window_size=None):
    # forecasted values, lags and differencing handled automatically
    forecaster = ForecasterAutoreg(
        regressor           = model,
        lags                = lags,
        differentiation     = differentiation,
        transformer_y       = scalar,            
        transformer_exog    = exog_scalar
    )

    # fit forecaster
    forecaster.fit(y=y_train, exog=X_train)

    y_val_pred = pd.Series()
    indexrange = range(y_val.index[0], y_val.index[-1] + 1, horizon)

    # perform forecast
    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # include max_lag last obs in train because of lags
        last_window = y_train.tail(forecaster.max_lag + 1)

        # include validation obs from first index to step before forecast
        if idx > 0:
            last_window = pd.concat([last_window, y_val.loc[:time-1]])
    
        # forecast starting from last window
        if useExog == True:
            preds = forecaster.predict(steps=horizon, exog=X_val.loc[time:time+horizon], last_window=last_window)
        else:
            preds = forecaster.predict(steps=horizon, exog=None, last_window=last_window)
        y_val_pred = pd.concat([y_val_pred, preds])

        print('Forecast iteration:', idx)
        clear_output(wait=True)
    print(timer.end())

    return model, y_val_pred


def fixedWindowWithoutLags(X_train, y_train, X_val, y_val, model, horizon, differentiation, scalar, exog_scalar, window_size=None):
    # predict
    X_val_transformed = pd.DataFrame(exog_scalar.transform(X_val), columns=X_val.columns)
    y_val_pred_transformed = model.predict(X_val_transformed).reshape(-1, 1)
    y_val_pred_diff = pd.Series(scalar.inverse_transform(y_val_pred_transformed).squeeze(1))
    y_val_pred_diff.index = y_val.index

    if differentiation == 1:
        y_val_pred = inverseDifferencing(y_val_pred_diff, y_val, horizon)
    else:
        y_val_pred = y_val_pred_diff.copy()

    return model, y_val_pred


"""
Expanding window methods
"""

def expandingWindowWithLags(X_train, y_train, X_val, y_val, model, 
                            horizon, differentiation, lags, 
                            scalar, exog_scalar, useExog, window_size=None):
    # initiate forecaster
    forecaster = ForecasterAutoreg(
        regressor           = model,
        lags                = lags,
        differentiation     = differentiation,
        transformer_y       = scalar,
        transformer_exog    = exog_scalar
        )

    y_val_pred = pd.Series()
    indexrange = range(y_val.index[0], y_val.index[-1] + 1, horizon)

    # define training data which will iteratively be expanded
    if useExog == True:
        X_expanding = X_train
    else:
        X_expanding=None

    y_expanding = y_train

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # include expanding window of previous validation data
        if idx > 0:
            y_expanding = pd.concat([y_expanding, y_val.loc[time-horizon:time-1]], ignore_index=True)
            if useExog == True:
                X_expanding = pd.concat([X_expanding, X_val.loc[time-horizon:time-1]], ignore_index=True)
    
        #print(y_expanding.index)

        forecaster.fit(y=y_expanding, exog=X_expanding)

        # forecast future validation data
        if useExog == True:
            X_horizon = X_val.loc[time:time+horizon]
        else:
            X_horizon = None
        
        preds = forecaster.predict(steps=horizon, exog=X_horizon)
        y_val_pred = pd.concat([y_val_pred, preds])

        print('Forecast iteration:', idx)
        clear_output(wait=True)

    print(timer.end())

    return model, y_val_pred


def expandingWindowWithoutLags(X_train, y_train, X_val, y_val, model, horizon, differentiation, scalar, exog_scalar, window_size=None):
    # setup
    y_val_pred_diff = pd.Series()
    indexrange = range(y_val.index[0], y_val.index[-1] + 1, horizon)

    # define training data which will iteratively be expanded
    X_expanding = X_train
    y_expanding = y_train

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # include expanding window of previous validation data
        if idx > 0:
            y_expanding = pd.concat([y_expanding, y_val.loc[time-horizon:time-1]], ignore_index=True)
            X_expanding = pd.concat([X_expanding, X_val.loc[time-horizon:time-1]], ignore_index=True)
    
        #print(y_expanding.index)

        # train model
        model, _, scalar, exog_scalar = trainModel(model=model, 
                                                   X_train=X_expanding, y_train=y_expanding, 
                                                   scalar=scalar, exog_scalar=exog_scalar, 
                                                   horizon=horizon, lags=None, differentiation=differentiation, 
                                                   useExog=True, useLags=False)
        
        # define exog horizon
        X_horizon = X_val.loc[time:time+horizon-1]
        y_horizon = y_val.loc[time:time+horizon-1]

        # make horizon prediction where output is differenced
        X_horizon_transformed = pd.DataFrame(exog_scalar.transform(X_horizon), columns=X_horizon.columns)
        y_horizon_pred_transformed = model.predict(X_horizon_transformed).reshape(-1, 1)
        y_horizon_pred_diff = pd.Series(scalar.inverse_transform(y_horizon_pred_transformed).squeeze(1))
        y_horizon_pred_diff.index = y_horizon.index

        y_val_pred_diff = pd.concat([y_val_pred_diff, y_horizon_pred_diff])

        print('Forecast iteration:', idx)
        clear_output(wait=True)

    print(timer.end())

    # inverse difference predictions
    if differentiation == 1:
        y_val_pred = inverseDifferencing(y_val_pred_diff, y_val, horizon)
    else:
        y_val_pred = y_val_pred_diff.copy()

    return model, y_val_pred


"""
Rolling window methods
"""

def rollingWindowWithLags(X_train, y_train, X_val, y_val, model, 
                            horizon, window_size, differentiation, lags, 
                            scalar, exog_scalar, useExog):
    # initiate forecaster
    forecaster = ForecasterAutoreg(
        regressor           = model,
        lags                = lags,
        differentiation     = differentiation,
        transformer_y       = scalar,
        transformer_exog    = exog_scalar
        )

    y_val_pred = pd.Series()
    indexrange = range(y_val.index[0], y_val.index[-1] + 1, horizon)

    # concat all train and val data so it can be subsetted on window
    if useExog == True:
        X_sliding = pd.concat([X_train, X_val])
    else:
        X_sliding = None

    y_sliding = pd.concat([y_train, y_val])

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # subset on window size
        if useExog == True:
            X_temp = X_sliding.loc[time - window_size : time-1]
        else:
            X_temp = None

        y_temp = y_sliding.loc[time - window_size : time-1]

        #print(y_temp.index)

        forecaster.fit(y=y_temp, exog=X_temp)

        # forecast future validation data
        if useExog == True:
            X_horizon = X_val.loc[time:time+horizon]
        else:
            X_horizon = None
        preds = forecaster.predict(steps=horizon, exog=X_horizon)

        y_val_pred = pd.concat([y_val_pred, preds])

        print('Forecast iteration:', idx)
        clear_output(wait=True)

    print(timer.end())

    return model, y_val_pred


def rollingWindowWithoutLags(X_train, y_train, X_val, y_val, model, 
                             horizon, window_size, differentiation, 
                             scalar, exog_scalar):
    y_val_pred_diff = pd.Series()
    indexrange = range(y_val.index[0], y_val.index[-1] + 1, horizon)

    # concat all train and val data so it can be subsetted on window
    X_sliding = pd.concat([X_train, X_val])
    y_sliding = pd.concat([y_train, y_val])

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # subset on window size
        X_temp = X_sliding.loc[time - window_size : time-1]
        y_temp = y_sliding.loc[time - window_size : time-1]

        #print(y_temp.index)

        # fit rolling model
        model_fitted, _, scalar, exog_scalar = trainModel(model=model, 
                                                          X_train=X_temp, y_train=y_temp, 
                                                          scalar=scalar, exog_scalar=exog_scalar, 
                                                          horizon=horizon, lags=None, differentiation=differentiation,
                                                          useExog=True, useLags=False)

        # forecast future validation data
        X_horizon = X_val.loc[time:time+horizon-1]
        y_horizon = y_val.loc[time:time+horizon-1]

        # make horizon prediction where output is differenced
        X_horizon_transformed = pd.DataFrame(exog_scalar.transform(X_horizon), columns=X_horizon.columns)
        y_horizon_pred_transformed = model.predict(X_horizon_transformed).reshape(-1, 1)
        y_horizon_pred_diff = pd.Series(scalar.inverse_transform(y_horizon_pred_transformed).squeeze(1))
        y_horizon_pred_diff.index = y_horizon.index

        y_val_pred_diff = pd.concat([y_val_pred_diff, y_horizon_pred_diff])

        print('Forecast iteration:', idx)
        clear_output(wait=True)

    print(timer.end())

    # inverse difference predictions
    if differentiation == 1:
        y_val_pred = inverseDifferencing(y_val_pred_diff, y_val, horizon)
    else:
        y_val_pred = y_val_pred_diff.copy()
    
    return model, y_val_pred


"""
Forecast method
"""

def sklearnForecast(X_train, y_train, X_val, y_val, model, 
                    horizon, differentiation, lags, use_exog, scalar, exog_scalar, 
                    window_type, window_size=None):
    
    if use_exog == False:
        X_train = None
        X_val = None

    useLags = False if lags is None else True
    useExog = False if X_train is None else True

    assert useLags == True or useExog == True, 'Must use lags and/or exogenous variables'
    assert useLags == True or useExog == True, 'Must use lags and/or exogenous variables'
    assert (X_train is None and X_val is None) or (X_train is not None and X_val is not None), 'X_train and X_val must both be real or None'
    assert differentiation is None or differentiation == 1, 'Differentiation must be None or 1'
    assert useLags == False or len(lags) > 0, 'List of lags cant be empty'

    if window_type == 'rolling':
        forecastWithLags = rollingWindowWithLags
        forecastWithoutLags = rollingWindowWithoutLags
        assert window_size is not None and type(window_size) is int and window_size > 0, 'Window size must be integer > 0'
    elif window_type == 'expanding':
        forecastWithLags = expandingWindowWithLags
        forecastWithoutLags = expandingWindowWithoutLags
        assert window_size is None, 'No window size for expanding window'
    elif window_type == 'fixed':
        forecastWithLags = fixedWindowWithLags
        forecastWithoutLags = fixedWindowWithoutLags
        assert window_size is None, 'No window size for fixed window'
    else:
        raise NotImplementedError
    

    # fit model to get training values
    model_fitted, y_train_pred, scalar, exog_scalar = trainModel(model=model, 
                                                                 X_train=X_train, y_train=y_train, 
                                                                 scalar=scalar, exog_scalar=exog_scalar, 
                                                                 horizon=horizon, lags=lags, differentiation=differentiation,
                                                                 useExog=useExog, useLags=useLags)

    if useLags == True:
        # with lags with or without exog
        model_fitted, y_val_pred = forecastWithLags(X_train=X_train, y_train=y_train, 
                                                         X_val=X_val, y_val=y_val, model=model, 
                                                         horizon=horizon, window_size=window_size, differentiation=differentiation, 
                                                         lags=lags, scalar=scalar, exog_scalar=exog_scalar, 
                                                         useExog=useExog)
    else:
        # without lags with exog
        model_fitted, y_val_pred = forecastWithoutLags(X_train=X_train, y_train=y_train, 
                                                            X_val=X_val, y_val=y_val,
                                                            model=model_fitted, horizon=horizon, 
                                                            window_size=window_size, differentiation=differentiation, 
                                                            scalar=scalar, exog_scalar=exog_scalar)

    y_train_pred = formatFittedValues(y_train_pred, y_train)
    y_val_pred = formatFittedValues(y_val_pred, y_val)
    return model, y_train_pred, y_val_pred

