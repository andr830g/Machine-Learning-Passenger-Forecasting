import pandas as pd

"""
Format continuous variable predictions to discrete variable predictions
"""

def formatFittedValues(y_pred, y_true):
    # round and cut off
    y_pred = y_pred.round()
    y_pred[y_pred < 0] = 0

    # re-index
    y_pred.index = y_true.index
    return y_pred


"""
Differencing and Inverse Differencing
"""

def difference(y_pred):
    y_pred_diff = y_pred.diff(periods=1).fillna(0)
    return y_pred_diff


def inverseDifferencing(y_pred_diff, y_true, horizon):
    y_pred = pd.Series()
    indexrange = range(y_true.index[0], y_true.index[-1] + 1, horizon)
    for idx, time in enumerate(indexrange):
        if idx == 0:
            constant = y_true.loc[time]
        else:
            constant = y_true.loc[time-1]
        preds = y_pred_diff.loc[time : time+horizon-1].cumsum() + constant
        y_pred = pd.concat([y_pred, preds])
    y_pred.index = y_true.index
    return y_pred