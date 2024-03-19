import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from utils.tools import *
from utils.metrics import Time

"""
Sklearn Forecasting
"""

def getFittedValues(model, X_train_temp, y_train_diff, scalar, exog_scalar):
    X_train_lagdiff_transformed = pd.DataFrame(exog_scalar.fit_transform(X_train_temp), columns=X_train_temp.columns)
    y_train_transformed = scalar.fit_transform(y_train_diff.to_numpy().reshape(-1, 1))
    model.fit(X_train_lagdiff_transformed, y_train_transformed)
    y_train_pred_transformed = model.predict(X_train_lagdiff_transformed).reshape(-1, 1)
    y_train_pred_diff = pd.Series(scalar.inverse_transform(y_train_pred_transformed).squeeze(1))
    return y_train_pred_diff


def fixedWindowWithLags(X_train, y_train, X_val, y_val, model, horizon, differentiation, lags, scalar, exog_scalar, useExog):
    ######
    # forecasted values, lags and differencing handled automatically
    ######
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
    
    print(timer.end())

    #######
    # fitted values
    #######
    # difference
    if differentiation == 1:
        y_train_diff = difference(y_train)
    else:
        y_train_diff = y_train.copy(deep=True)
    
    # add lagged values
    if useExog == True:
        X_train_temp = X_train.copy(deep=True)
    else:
        X_train_temp = pd.DataFrame(index=y_train.index)
    
    if lags:
        for lag in lags:
            X_train_temp[f'lag_{lag}'] = y_train_diff.shift(periods=lag).fillna(0)

    # fit differenced and transformed
    y_train_pred_diff = getFittedValues(model, X_train_temp, y_train_diff, scalar, exog_scalar)

    # inverse difference
    if differentiation == 1:
        y_train_pred = inverseDifferencing(y_train_pred_diff, y_train, horizon)
    else:
        y_train_pred = y_train_pred_diff.copy(deep=True)

    return y_train_pred, y_val_pred


def fixedWindowWithoutLags(X_train, y_train, X_val, y_val, model, horizon, differentiation, scalar, exog_scalar):
    # diff
    if differentiation == 1:
        y_train_diff = difference(y_train)
    else:
        y_train_diff = y_train.copy()

    # fit differenced and transformed
    y_train_pred_diff = getFittedValues(model, X_train, y_train_diff, scalar, exog_scalar)
    
    # predict
    X_val_transformed = pd.DataFrame(exog_scalar.transform(X_val), columns=X_val.columns)
    y_val_pred_transformed = model.predict(X_val_transformed).reshape(-1, 1)
    y_val_pred_diff = pd.Series(scalar.inverse_transform(y_val_pred_transformed).squeeze(1))
    y_val_pred_diff.index = y_val.index

    # inverse diff
    if differentiation == 1:
        y_train_pred = inverseDifferencing(y_train_pred_diff, y_train, horizon)
        y_val_pred = inverseDifferencing(y_val_pred_diff, y_val, horizon)
    else:
        y_train_pred = y_train_pred_diff.copy()
        y_val_pred = y_val_pred_diff.copy()

    return y_train_pred, y_val_pred


def fixedWindowForecastSklearn(X_train, y_train, X_val, y_val, model, 
                               horizon, differentiation=None, lags=None, scalar=None, exog_scalar=None,
                               useLags=True, useExog=True):
    useLags = False if lags is None else True
    useExog = False if X_train is None else True
    assert useLags == True or useExog == True, 'must use lags and/or exogenous variables'

    if useLags == True:
        # with lags with or without exog
        y_train_pred, y_val_pred = fixedWindowWithLags(X_train, y_train, X_val, y_val, model, 
                                         horizon, differentiation, lags, 
                                         scalar, exog_scalar, useExog)
    else:
        # without lags with exog
        y_train_pred, y_val_pred = fixedWindowWithoutLags(X_train, y_train, X_val, y_val, model, 
                                                          horizon, differentiation, scalar, exog_scalar)

    y_train_pred = formatFittedValues(y_train_pred, y_train)
    y_val_pred = formatFittedValues(y_val_pred, y_val)
    return model, y_train_pred, y_val_pred


def expandingWindowForecastSklearn(X_train, y_train, X_val, y_val, model, 
                                   horizon, differentiation=None, lags=None, scalar=None, exog_scalar=None):
    useLags = False if lags is None else True
    useExog = False if X_train is None else True
    assert useLags == True or useExog == True, 'must use lags and/or exogenous variables'
    if useLags == False or useExog == False:
        raise NotImplementedError

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
    X_expanding = X_train
    y_expanding = y_train

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # include expanding window of previous validation data
        if idx > 0:
            y_expanding = pd.concat([y_expanding, y_val.loc[time-horizon:time-1]], ignore_index=True)
            X_expanding = pd.concat([X_expanding, X_val.loc[time-horizon:time-1]], ignore_index=True)
    
        print(y_expanding.index)

        forecaster.fit(y=y_expanding, exog=X_expanding)

        # forecast future validation data
        preds = forecaster.predict(steps=horizon, exog=X_val.loc[time:time+horizon])
        y_val_pred = pd.concat([y_val_pred, preds])
    print(timer.end())

    # fit model to get fitted training values
    X_train_temp = X_train.copy(deep=True)
    if lags:
        for lag in lags:
            X_train_temp[f'lag_{lag}'] = y_train.shift(periods=lag).fillna(0)
    
    X_train_temp_transformed = pd.DataFrame(exog_scalar.fit_transform(X_train_temp), columns=X_train_temp.columns)
    y_train_temp_transformed = scalar.fit_transform(y_train.to_numpy().reshape(-1, 1))
    model.fit(X_train_temp_transformed, y_train_temp_transformed)
    y_train_pred_transformed = model.predict(X_train_temp_transformed).reshape(-1, 1)
    y_train_pred = pd.Series(scalar.inverse_transform(y_train_pred_transformed).squeeze(1))

    y_train_pred = formatFittedValues(y_train_pred, y_train)
    y_val_pred = formatFittedValues(y_val_pred, y_val)
    return model, y_train_pred, y_val_pred


def rollingWindowForecastSklearn(X_train, y_train, X_val, y_val, model, 
                                 horizon, window_size, differentiation=None, lags=None, scalar=None, exog_scalar=None):
    useLags = False if lags is None else True
    useExog = False if X_train is None else True
    assert useLags == True or useExog == True, 'must use lags and/or exogenous variables'
    if useLags == False or useExog == False:
        raise NotImplementedError

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
    X_sliding = pd.concat([X_train, X_val])
    y_sliding = pd.concat([y_train, y_val])

    timer = Time()
    timer.start()
    for idx, time in enumerate(indexrange):
        # subset on window size
        X_temp = X_sliding.loc[time - window_size : time-1]
        y_temp = y_sliding.loc[time - window_size : time-1]

        print(y_temp.index)

        forecaster.fit(y=y_temp, exog=X_temp)

        # forecast future validation data
        preds = forecaster.predict(steps=horizon, exog=X_val.loc[time:time+horizon])
        y_val_pred = pd.concat([y_val_pred, preds])
    print(timer.end())

    # fit model to get fitted training values
    X_train_temp = X_train.copy(deep=True)
    if lags:
        for lag in lags:
            X_train_temp[f'lag_{lag}'] = y_train.shift(periods=lag).fillna(0)
    
    X_train_temp_transformed = pd.DataFrame(exog_scalar.fit_transform(X_train_temp), columns=X_train_temp.columns)
    y_train_temp_transformed = scalar.fit_transform(y_train.to_numpy().reshape(-1, 1))
    model.fit(X_train_temp_transformed, y_train_temp_transformed)
    y_train_pred_transformed = model.predict(X_train_temp_transformed).reshape(-1, 1)
    y_train_pred = pd.Series(scalar.inverse_transform(y_train_pred_transformed).squeeze(1))

    y_train_pred = formatFittedValues(y_train_pred, y_train)
    y_val_pred = formatFittedValues(y_val_pred, y_val)
    return model, y_train_pred, y_val_pred