from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import time

def MAE(y, yhat):
    """
    Mean Absolute Error in % is MAE/mean(y)
    y is ground truth and yhat is prediction
    """
    mae = mean_absolute_error(y, yhat)
    meany = np.mean(y)
    maepercent = mae/meany
    return f'MAE%: {maepercent:.3f}'


def MAPE(y, yhat):
    """
    Mean Absolute Percentage Error
    """
    mape = mean_absolute_percentage_error(y, yhat)
    return f'MAPE: {mape:.3f}'



def RMSE(y, yhat):
    """
    Root Mean Squared Error in % is RMSE/mean(y)
    y is ground truth and yhat is prediction
    """
    rmse = np.sqrt(mean_squared_error(y, yhat))
    meany = np.mean(y)
    rmsepercent = rmse/meany
    return f'RMSE%: {rmsepercent:.3f}'


class Time:
    """
    Class for starting and ending execution time
    """
    def __init__(self):
        self.startTime = None
        self.endTime = None

    def start(self):
        self.startTime = time.time()

    def end(self):
        assert self.startTime != None, 'Must start timer before ending timer'

        self.endTime = time.time()

        return f'Execution time: {self.endTime - self.startTime:.3f} sec'
