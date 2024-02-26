import pandas as pd

def getTrainData(agglevel=60, train_test_split='2023-10-01', start_date='2021-06-27', hours_to_exclude=[1, 2, 3, 4]):
    assert start_date <= train_test_split, 'Invalid days'

    path = 'data/'
    file = f'data{agglevel}.csv'
    df = pd.read_csv(path + file, sep=',')
    df = df.sort_values(by='datetime')
    df = df[(df['date'] >= start_date) & (df['date'] < train_test_split)]

    df = excludeHours(df, hours_to_exclude)
    df = addFeatures(df)

    return df

def getTestData(agglevel=60, train_test_split='2023-10-01', end_date='2023-12-31', hours_to_exclude=[1, 2, 3, 4]):
    assert train_test_split <= end_date, 'Invalid days'

    path = 'data/'
    file = f'data{agglevel}.csv'
    df = pd.read_csv(path + file, sep=',')
    df = df.sort_values(by='datetime')
    df = df[(df['date'] >= train_test_split) & (df['date'] <= end_date)]

    df = excludeHours(df, hours_to_exclude)
    df = addFeatures(df)

    return df

def performTrainValSplit(df, train_val_split=''):
    raise NotImplementedError
    #return df_train, df_val

def excludeHours(df, hours_to_exclude):
    df = df[~df['hour'].isin(hours_to_exclude)]
    return df

def addFeatures(df):
    return df
