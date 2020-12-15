import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics
import gc
gc.enable()
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor, BayesianRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
import time
np.random.seed(1343)
start_time = time.time()
if 'start_time' not in TANGSHAN:
    import csv
    if isinstance(start_time, np.ndarray) or isinstance(start_time, pd.
        DataFrame) or isinstance(start_time, pd.Series):
        shape_size = start_time.shape
    elif isinstance(start_time, list):
        shape_size = len(start_time)
    else:
        shape_size = 'any'
    check_type = type(start_time)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('start_time')
        writer = csv.writer(f)
        writer.writerow(['start_time', 20, check_type, shape_size])
tcurrent = start_time
if 'tcurrent' not in TANGSHAN:
    import csv
    if isinstance(tcurrent, np.ndarray) or isinstance(tcurrent, pd.DataFrame
        ) or isinstance(tcurrent, pd.Series):
        shape_size = tcurrent.shape
    elif isinstance(tcurrent, list):
        shape_size = len(tcurrent)
    else:
        shape_size = 'any'
    check_type = type(tcurrent)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tcurrent')
        if 'str' not in TANGSHAN:
            import csv
            if isinstance(str, np.ndarray) or isinstance(str, pd.DataFrame
                ) or isinstance(str, pd.Series):
                shape_size = str.shape
            elif isinstance(str, list):
                shape_size = len(str)
            else:
                shape_size = 'any'
            check_type = type(str)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('str')
                writer = csv.writer(f)
                writer.writerow(['str', 24, check_type, shape_size])
        writer = csv.writer(f)
        if 'dtypes' not in TANGSHAN:
            import csv
            if isinstance(dtypes, np.ndarray) or isinstance(dtypes, pd.
                DataFrame) or isinstance(dtypes, pd.Series):
                shape_size = dtypes.shape
            elif isinstance(dtypes, list):
                shape_size = len(dtypes)
            else:
                shape_size = 'any'
            check_type = type(dtypes)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('dtypes')
                writer = csv.writer(f)
                writer.writerow(['dtypes', 25, check_type, shape_size])
        writer.writerow(['tcurrent', 21, check_type, shape_size])
dtypes = {'id': 'int64', 'item_nbr': 'int32', 'store_nbr': 'int8',
    'onpromotion': str}
if 'str' not in TANGSHAN:
    import csv
    if isinstance(str, np.ndarray) or isinstance(str, pd.DataFrame
        ) or isinstance(str, pd.Series):
        shape_size = str.shape
    elif isinstance(str, list):
        shape_size = len(str)
    else:
        shape_size = 'any'
    check_type = type(str)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('str')
        writer = csv.writer(f)
        writer.writerow(['str', 24, check_type, shape_size])
data = {'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=
    ['date']), 'tes': pd.read_csv('../input/test.csv', dtype=dtypes,
    parse_dates=['date']), 'ite': pd.read_csv('../input/items.csv'), 'sto':
    pd.read_csv('../input/stores.csv'), 'trn': pd.read_csv(
    '../input/transactions.csv', parse_dates=['date']), 'hol': pd.read_csv(
    '../input/holidays_events.csv', dtype={'transferred': str}, parse_dates
    =['date']), 'oil': pd.read_csv('../input/oil.csv', parse_dates=['date'])}
if 'dtypes' not in TANGSHAN:
    import csv
    if isinstance(dtypes, np.ndarray) or isinstance(dtypes, pd.DataFrame
        ) or isinstance(dtypes, pd.Series):
        shape_size = dtypes.shape
    elif isinstance(dtypes, list):
        shape_size = len(dtypes)
    else:
        shape_size = 'any'
    check_type = type(dtypes)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dtypes')
        writer = csv.writer(f)
        writer.writerow(['dtypes', 25, check_type, shape_size])
print('Datasets processing')
train = data['tra'][data['tra']['date'].dt.year >= 2016]
del data['tra']
gc.collect()
target = train['unit_sales'].values
target[target < 0.0] = 0.0
train['unit_sales'] = np.log1p(target)


def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df


def df_transform(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({(0): 1.0, (1): 1.25})
    df = df.fillna(-1)
    return df


data['ite'] = df_lbl_enc(data['ite'])
train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
del data['tes']
gc.collect()
del data['ite']
gc.collect()
train = pd.merge(train, data['trn'], how='left', on=['date', 'store_nbr'])
test = pd.merge(test, data['trn'], how='left', on=['date', 'store_nbr'])
del data['trn']
gc.collect()
target = train['transactions'].values
target[target < 0.0] = 0.0
train['transactions'] = np.log1p(target)
if 'target' not in TANGSHAN:
    import csv
    if isinstance(target, np.ndarray) or isinstance(target, pd.DataFrame
        ) or isinstance(target, pd.Series):
        shape_size = target.shape
    elif isinstance(target, list):
        shape_size = len(target)
    else:
        shape_size = 'any'
    check_type = type(target)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('target')
        writer = csv.writer(f)
        writer.writerow(['target', 69, check_type, shape_size])
data['sto'] = df_lbl_enc(data['sto'])
train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])
test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])
if 'test' not in TANGSHAN:
    import csv
    if isinstance(test, np.ndarray) or isinstance(test, pd.DataFrame
        ) or isinstance(test, pd.Series):
        shape_size = test.shape
    elif isinstance(test, list):
        shape_size = len(test)
    else:
        shape_size = 'any'
    check_type = type(test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test')
        writer = csv.writer(f)
        writer.writerow(['test', 73, check_type, shape_size])
if 'data' not in TANGSHAN:
    import csv
    if isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame
        ) or isinstance(data, pd.Series):
        shape_size = data.shape
    elif isinstance(data, list):
        shape_size = len(data)
    else:
        shape_size = 'any'
    check_type = type(data)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('data')
        writer = csv.writer(f)
        writer.writerow(['data', 73, check_type, shape_size])
del data['sto']
gc.collect()
data['hol'] = data['hol'][data['hol']['locale'] == 'National'][['date',
    'transferred']]
data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0,
    'True': 1})
train = pd.merge(train, data['hol'], how='left', on=['date'])
test = pd.merge(test, data['hol'], how='left', on=['date'])
del data['hol']
gc.collect()
train = pd.merge(train, data['oil'], how='left', on=['date'])
if 'train' not in TANGSHAN:
    import csv
    if isinstance(train, np.ndarray) or isinstance(train, pd.DataFrame
        ) or isinstance(train, pd.Series):
        shape_size = train.shape
    elif isinstance(train, list):
        shape_size = len(train)
    else:
        shape_size = 'any'
    check_type = type(train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train')
        writer = csv.writer(f)
        writer.writerow(['train', 82, check_type, shape_size])
test = pd.merge(test, data['oil'], how='left', on=['date'])
del data['oil']
gc.collect()
train = df_transform(train)
test = df_transform(test)
col = [c for c in train if c not in ['id', 'date', 'unit_sales']]
if 'c' not in TANGSHAN:
    import csv
    if isinstance(c, np.ndarray) or isinstance(c, pd.DataFrame) or isinstance(c
        , pd.Series):
        shape_size = c.shape
    elif isinstance(c, list):
        shape_size = len(c)
    else:
        shape_size = 'any'
    check_type = type(c)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('c')
        writer = csv.writer(f)
        writer.writerow(['c', 88, check_type, shape_size])
x1 = train[(train['yea'] != 2016) & (train['mon'] != 8)][col]
if 'x1' not in TANGSHAN:
    import csv
    if isinstance(x1, np.ndarray) or isinstance(x1, pd.DataFrame
        ) or isinstance(x1, pd.Series):
        shape_size = x1.shape
    elif isinstance(x1, list):
        shape_size = len(x1)
    else:
        shape_size = 'any'
    check_type = type(x1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x1')
        writer = csv.writer(f)
        writer.writerow(['x1', 92, check_type, shape_size])
if 'col' not in TANGSHAN:
    import csv
    if isinstance(col, np.ndarray) or isinstance(col, pd.DataFrame
        ) or isinstance(col, pd.Series):
        shape_size = col.shape
    elif isinstance(col, list):
        shape_size = len(col)
    else:
        shape_size = 'any'
    check_type = type(col)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('col')
        writer = csv.writer(f)
        writer.writerow(['col', 92, check_type, shape_size])
x2 = train[(train['yea'] == 2016) & (train['mon'] == 8)][col]
y1 = train[(train['yea'] != 2016) & (train['mon'] != 8)]['unit_sales'].values
y2 = train[(train['yea'] == 2016) & (train['mon'] == 8)]['unit_sales'].values
del train
gc.collect()


def NWRMSLE(preds, train_data):
    return 'nwrmsle', mean_squared_log_error(train_data.get_label(), preds
        ), False


import lightgbm as lgb
lgb_train = lgb.Dataset(x1, y1)
if 'lgb_train' not in TANGSHAN:
    import csv
    if isinstance(lgb_train, np.ndarray) or isinstance(lgb_train, pd.DataFrame
        ) or isinstance(lgb_train, pd.Series):
        shape_size = lgb_train.shape
    elif isinstance(lgb_train, list):
        shape_size = len(lgb_train)
    else:
        shape_size = 'any'
    check_type = type(lgb_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('lgb_train')
        writer = csv.writer(f)
        writer.writerow(['lgb_train', 107, check_type, shape_size])
if 'y1' not in TANGSHAN:
    import csv
    if isinstance(y1, np.ndarray) or isinstance(y1, pd.DataFrame
        ) or isinstance(y1, pd.Series):
        shape_size = y1.shape
    elif isinstance(y1, list):
        shape_size = len(y1)
    else:
        shape_size = 'any'
    check_type = type(y1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y1')
        writer = csv.writer(f)
        writer.writerow(['y1', 107, check_type, shape_size])
lgb_eval = lgb.Dataset(x2, y2, reference=lgb_train)
if 'y2' not in TANGSHAN:
    import csv
    if isinstance(y2, np.ndarray) or isinstance(y2, pd.DataFrame
        ) or isinstance(y2, pd.Series):
        shape_size = y2.shape
    elif isinstance(y2, list):
        shape_size = len(y2)
    else:
        shape_size = 'any'
    check_type = type(y2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y2')
        writer = csv.writer(f)
        writer.writerow(['y2', 108, check_type, shape_size])
if 'lgb_eval' not in TANGSHAN:
    import csv
    if isinstance(lgb_eval, np.ndarray) or isinstance(lgb_eval, pd.DataFrame
        ) or isinstance(lgb_eval, pd.Series):
        shape_size = lgb_eval.shape
    elif isinstance(lgb_eval, list):
        shape_size = len(lgb_eval)
    else:
        shape_size = 'any'
    check_type = type(lgb_eval)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('lgb_eval')
        writer = csv.writer(f)
        writer.writerow(['lgb_eval', 108, check_type, shape_size])
if 'x2' not in TANGSHAN:
    import csv
    if isinstance(x2, np.ndarray) or isinstance(x2, pd.DataFrame
        ) or isinstance(x2, pd.Series):
        shape_size = x2.shape
    elif isinstance(x2, list):
        shape_size = len(x2)
    else:
        shape_size = 'any'
    check_type = type(x2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x2')
        writer = csv.writer(f)
        writer.writerow(['x2', 108, check_type, shape_size])
mean_squared_log_error(np.exp(lgb_train.get_label()), np.exp(lgb_train.
    get_label()))
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'poisson',
    'metric': 'NWRMSLE', 'num_leaves': 50, 'learning_rate': 0.1,
    'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': 0}
print('Start training...')
evals_result = {}
if 'evals_result' not in TANGSHAN:
    import csv
    if isinstance(evals_result, np.ndarray) or isinstance(evals_result, pd.
        DataFrame) or isinstance(evals_result, pd.Series):
        shape_size = evals_result.shape
    elif isinstance(evals_result, list):
        shape_size = len(evals_result)
    else:
        shape_size = 'any'
    check_type = type(evals_result)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('evals_result')
        writer = csv.writer(f)
        writer.writerow(['evals_result', 128, check_type, shape_size])
gbm = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=(
    lgb_train, lgb_eval), verbose_eval=50, feval=NWRMSLE, evals_result=
    evals_result, early_stopping_rounds=20)
if 'params' not in TANGSHAN:
    import csv
    if isinstance(params, np.ndarray) or isinstance(params, pd.DataFrame
        ) or isinstance(params, pd.Series):
        shape_size = params.shape
    elif isinstance(params, list):
        shape_size = len(params)
    else:
        shape_size = 'any'
    check_type = type(params)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('params')
        writer = csv.writer(f)
        writer.writerow(['params', 129, check_type, shape_size])
print('Save model...')
if 'print' not in TANGSHAN:
    import csv
    if isinstance(print, np.ndarray) or isinstance(print, pd.DataFrame
        ) or isinstance(print, pd.Series):
        shape_size = print.shape
    elif isinstance(print, list):
        shape_size = len(print)
    else:
        shape_size = 'any'
    check_type = type(print)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('print')
        writer = csv.writer(f)
        writer.writerow(['print', 138, check_type, shape_size])
gbm.save_model('model.txt')
if 'gbm' not in TANGSHAN:
    import csv
    if isinstance(gbm, np.ndarray) or isinstance(gbm, pd.DataFrame
        ) or isinstance(gbm, pd.Series):
        shape_size = gbm.shape
    elif isinstance(gbm, list):
        shape_size = len(gbm)
    else:
        shape_size = 'any'
    check_type = type(gbm)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('gbm')
        writer = csv.writer(f)
        writer.writerow(['gbm', 144, check_type, shape_size])
print('Start predicting...')
y_pred = gbm.predict(test[col].values, num_iteration=gbm.best_iteration)
if 'y_pred' not in TANGSHAN:
    import csv
    if isinstance(y_pred, np.ndarray) or isinstance(y_pred, pd.DataFrame
        ) or isinstance(y_pred, pd.Series):
        shape_size = y_pred.shape
    elif isinstance(y_pred, list):
        shape_size = len(y_pred)
    else:
        shape_size = 'any'
    check_type = type(y_pred)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_pred')
        writer = csv.writer(f)
        writer.writerow(['y_pred', 148, check_type, shape_size])
y_pred[0:5]
sub = pd.DataFrame(test['id'])
sub['unit_sales'] = 0
if 'sub' not in TANGSHAN:
    import csv
    if isinstance(sub, np.ndarray) or isinstance(sub, pd.DataFrame
        ) or isinstance(sub, pd.Series):
        shape_size = sub.shape
    elif isinstance(sub, list):
        shape_size = len(sub)
    else:
        shape_size = 'any'
    check_type = type(sub)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('sub')
        writer = csv.writer(f)
        writer.writerow(['sub', 152, check_type, shape_size])
sub['unit_sales'] = y_pred
sub.loc[sub['unit_sales'] < 0, ['unit_sales']] = 0
sub.to_csv('subm01.csv.gz', index=False, float_format='%.3f', compression=
    'gzip')
