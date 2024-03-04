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
    return np.round(maepercent, decimals=3)


def MAPE(y, yhat):
    """
    Mean Absolute Percentage Error
    y is ground truth and yhat is prediction
    """
    #mape = mean_absolute_percentage_error(y, yhat)
    ape = np.abs(yhat - y)/y
    ape = ape.replace([-np.inf, np.inf], 0)
    mape = np.mean(ape)
    return np.round(mape, decimals=3)



def RMSE(y, yhat):
    """
    Root Mean Squared Error in % is RMSE/mean(y)
    y is ground truth and yhat is prediction
    """
    rmse = np.sqrt(mean_squared_error(y, yhat))
    meany = np.mean(y)
    rmsepercent = rmse/meany
    return np.round(rmsepercent, decimals=3)


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
