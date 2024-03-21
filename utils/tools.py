import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import *

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
    y_pred = pd.Series(dtype=float)
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


"""
Plotting
"""

def plotFitAndPredictions(y_train_pred, y_val_pred, y_train_true, y_val_true, trainDateCol=[], valDateCol=[], dates=True, print_accuracy=False):
    assert (len(trainDateCol) > 0 and len(valDateCol) > 0) or not dates, 'must input trainDateCol and valDateCol or else set dates=False'

    text_constant = 19
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.set_tight_layout('h_pad')

    ax[0].plot(y_train_true.index, y_train_true, color='red', label='gt')
    ax[0].plot(y_train_true.index, y_train_pred, color='blue', alpha=0.5, label='fitted')
    ax[0].set_title('Fitted Training Data')
    ax[0].legend(loc='upper right')
    ax[0].set_xlim(y_train_true.index.start, y_train_true.index.stop)
    ax[0].set_ylim(np.min(y_train_true), np.max(y_train_true) + np.max(y_train_true)//6)
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Passenger Count')

    ax[1].plot(y_val_true.index, y_val_true, color='red', label='gt')
    ax[1].plot(y_val_true.index, y_val_pred, color='blue', alpha=0.5, label='pred')
    ax[1].set_title('Predicted Validation Data')
    ax[1].legend(loc='upper right')
    ax[1].set_xlim(y_val_true.index.start, y_val_true.index.stop)
    ax[1].set_ylim(np.min(y_val_true), np.max(y_val_true) + np.max(y_val_true)//6)
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Passenger Count')

    if dates:
        text_constant = 28
        
        dateInterval = len(y_train_true)//365*24
        ax[0].set_xticks([i for i in range(y_train_true.index.start, y_train_true.index.stop, dateInterval)])
        ax[0].set_xticklabels(trainDateCol[::dateInterval], rotation=90)

        dateInterval = len(y_val_true)//365*24
        ax[1].set_xticks([i for i in range(y_val_true.index.start, y_val_true.index.stop, dateInterval)])
        ax[1].set_xticklabels(valDateCol[::dateInterval], rotation=90)

    ax[0].text(y_train_true.index.start + (y_train_true.index.stop - y_train_true.index.start)//54, np.max(y_train_true) + np.max(y_train_true)//text_constant,
               f'Train MAE%: {MAE(y=y_train_true, yhat=y_train_pred)}\n'
               + f'Train MAPE: {MAPE(y=y_train_true, yhat=y_train_pred)}\n'
               + f'Train RMSE%: {RMSE(y=y_train_true, yhat=y_train_pred)}',
               bbox=dict(facecolor='white', alpha=0.5),
               fontsize=9)
    ax[1].text(y_val_true.index.start + (y_val_true.index.stop - y_val_true.index.start)//54, np.max(y_val_true) + np.max(y_val_true)//text_constant,
               f'Val MAE%: {MAE(y=y_val_true, yhat=y_val_pred)}\n'
               + f'Val MAPE: {MAPE(y=y_val_true, yhat=y_val_pred)}\n'
               + f'Val RMSE%: {RMSE(y=y_val_true, yhat=y_val_pred)}',
               bbox=dict(facecolor='white', alpha=0.5),
               fontsize=9)
    
    plt.show()

    if print_accuracy:
        print('Train MAE%:', MAE(y=y_train_true, yhat=y_train_pred))
        print('Train MAPE:', MAPE(y=y_train_true, yhat=y_train_pred))
        print('Train RMSE%:', RMSE(y=y_train_true, yhat=y_train_pred))
        print('---')
        print('Val MAE%:', MAE(y=y_val_true, yhat=y_val_pred))
        print('Val MAPE:', MAPE(y=y_val_true, yhat=y_val_pred))
        print('Val RMSE%:', RMSE(y=y_val_true, yhat=y_val_pred))


def plotLossCurves(train_loss_list, val_loss_list, epoch_range):
    plt.plot(epoch_range, train_loss_list, color='blue', label='Training Loss')
    plt.plot(epoch_range, val_loss_list, color='orange', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()