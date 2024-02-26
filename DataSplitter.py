import pandas as pd

def getTrainData(agglevel=60, train_test_split='2023-06-01', start_date='2021-06-27'):
    assert start_date <= train_test_split, 'Invalid days'

    path = 'data/'
    file = f'data{agglevel}.csv'
    df = pd.read_csv(path + file, sep=',')
    df = df.sort_values(by='datetime')
    df = addFeatures(df)
    train = df[(df['date'] >= start_date) & (df['date'] < train_test_split)]
    return train

def getTestData(agglevel=60, train_test_split='2023-06-01', end_date='2023-12-31'):
    assert train_test_split <= end_date, 'Invalid days'

    path = 'data/'
    file = f'data{agglevel}.csv'
    df = pd.read_csv(path + file, sep=',')
    df = df.sort_values(by='datetime')
    df = addFeatures(df)
    test = df[(df['date'] >= train_test_split) & (df['date'] <= end_date)]
    return test

def performTrainValSplit(df, train_val_split=''):
    raise NotImplementedError
    #return df_train, df_val


def addFeatures(df):
    return df