import pandas as pd

def readData(agglevel):
    path = 'data/'
    file = f'data{agglevel}.csv'
    df = pd.read_csv(path + file, sep=',')
    df = df.sort_values(by=['line', 'datetime'])
    return df


def getTrainData(agglevel=60, train_test_split='2023-10-01', start_date='2021-06-27', hours_to_exclude=[1, 2, 3, 4]):
    assert start_date <= train_test_split, 'Invalid days'

    df = readData(agglevel)
    df = excludeHours(df, hours_to_exclude)
    #df = addFeatures(df, agglevel)

    df = df[(df['date'] >= start_date) & (df['date'] < train_test_split)]
    return df


def getTestData(agglevel=60, train_test_split='2023-10-01', end_date='2023-12-31', hours_to_exclude=[1, 2, 3, 4]):
    assert train_test_split <= end_date, 'Invalid days'

    df = readData(agglevel)
    df = excludeHours(df, hours_to_exclude)
    #df = addFeatures(df, agglevel)

    df = df[(df['date'] >= train_test_split) & (df['date'] <= end_date)]
    return df


def performTrainValSplit(df, train_val_split='2023-07-01'):

    df_train = df[df['date'] < train_val_split]
    df_val = df[df['date'] >= train_val_split]

    return df_train, df_val


def excludeHours(df, hours_to_exclude):

    df = df[~df['hour'].isin(hours_to_exclude)]
    
    return df


def addFeatures(df, agglevel):
    # 1-difference
    df['diff'] = df['passengersBoarding'].diff(periods=1)
    
    if agglevel == 60:
        # last 19 hours
        for i in range(1, 19+1):
            df[f'diffLag{i}'] = df['diff'].shift(periods=i, fill_value=0)

        # daily lags within 7 days
        for i in range(1, 7+1):
            df[f'diffLag{20*i}'] = df['diff'].shift(periods=20*i, fill_value=0)
        
    elif agglevel == 30:
        # last 39 30-minutes (19.5 hours)
        for i in range(1, 39+1):
            df[f'diffLag{i}'] = df['diff'].shift(periods=i, fill_value=0)
        
        # daily lags within 7 days
        for i in range(1, 7+1):
            df[f'diffLag{40*i}'] = df['diff'].shift(periods=40*i, fill_value=0)
        
    elif agglevel == 15:
        # last 79 15-minutes (19.75 hours)
        for i in range(1, 79+1):
            df[f'diffLag{i}'] = df['diff'].shift(periods=i, fill_value=0)
        
        # daily lags within 7 days
        for i in range(1, 7+1):
            df[f'diffLag{80*i}'] = df['diff'].shift(periods=80*i, fill_value=0)
    
    return df