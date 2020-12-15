import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
homesite_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
homesite_df.head()
homesite_df.info()
print('----------------------------')
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
        writer.writerow(['print', 53, check_type, shape_size])
test_df.info()
homesite_df = homesite_df.drop(['QuoteNumber'], axis=1)
if 'homesite_df' not in TANGSHAN:
    import csv
    if isinstance(homesite_df, np.ndarray) or isinstance(homesite_df, pd.
        DataFrame) or isinstance(homesite_df, pd.Series):
        shape_size = homesite_df.shape
    elif isinstance(homesite_df, list):
        shape_size = len(homesite_df)
    else:
        shape_size = 'any'
    check_type = type(homesite_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('homesite_df')
        writer = csv.writer(f)
        writer.writerow(['homesite_df', 59, check_type, shape_size])
homesite_df.drop(['Original_Quote_Date'], axis=1, inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1, inplace=True)
sns.countplot(x='QuoteConversion_Flag', data=homesite_df)
homesite_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)
if 'test_df' not in TANGSHAN:
    import csv
    if isinstance(test_df, np.ndarray) or isinstance(test_df, pd.DataFrame
        ) or isinstance(test_df, pd.Series):
        shape_size = test_df.shape
    elif isinstance(test_df, list):
        shape_size = len(test_df)
    else:
        shape_size = 'any'
    check_type = type(test_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test_df')
        writer = csv.writer(f)
        writer.writerow(['test_df', 75, check_type, shape_size])
from sklearn import preprocessing
for f in homesite_df.columns:
    if homesite_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        if 'lbl' not in TANGSHAN:
            import csv
            if isinstance(lbl, np.ndarray) or isinstance(lbl, pd.DataFrame
                ) or isinstance(lbl, pd.Series):
                shape_size = lbl.shape
            elif isinstance(lbl, list):
                shape_size = len(lbl)
            else:
                shape_size = 'any'
            check_type = type(lbl)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('lbl')
                writer = csv.writer(f)
                writer.writerow(['lbl', 87, check_type, shape_size])
        lbl.fit(np.unique(list(homesite_df[f].values) + list(test_df[f].
            values)))
        homesite_df[f] = lbl.transform(list(homesite_df[f].values))
        if 'list' not in TANGSHAN:
            import csv
            if isinstance(list, np.ndarray) or isinstance(list, pd.DataFrame
                ) or isinstance(list, pd.Series):
                shape_size = list.shape
            elif isinstance(list, list):
                shape_size = len(list)
            else:
                shape_size = 'any'
            check_type = type(list)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('list')
                writer = csv.writer(f)
                writer.writerow(['list', 91, check_type, shape_size])
        test_df[f] = lbl.transform(list(test_df[f].values))
        if 'f' not in TANGSHAN:
            import csv
            if isinstance(f, np.ndarray) or isinstance(f, pd.DataFrame
                ) or isinstance(f, pd.Series):
                shape_size = f.shape
            elif isinstance(f, list):
                shape_size = len(f)
            else:
                shape_size = 'any'
            check_type = type(f)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('f')
                writer = csv.writer(f)
                writer.writerow(['f', 93, check_type, shape_size])
X_train = homesite_df.drop('QuoteConversion_Flag', axis=1)
Y_train = homesite_df['QuoteConversion_Flag']
X_test = test_df.drop('QuoteNumber', axis=1).copy()
params = {'objective': 'binary:logistic'}
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
        writer.writerow(['params', 105, check_type, shape_size])
T_train_xgb = xgb.DMatrix(X_train, Y_train)
if 'T_train_xgb' not in TANGSHAN:
    import csv
    if isinstance(T_train_xgb, np.ndarray) or isinstance(T_train_xgb, pd.
        DataFrame) or isinstance(T_train_xgb, pd.Series):
        shape_size = T_train_xgb.shape
    elif isinstance(T_train_xgb, list):
        shape_size = len(T_train_xgb)
    else:
        shape_size = 'any'
    check_type = type(T_train_xgb)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('T_train_xgb')
        writer = csv.writer(f)
        writer.writerow(['T_train_xgb', 109, check_type, shape_size])
if 'Y_train' not in TANGSHAN:
    import csv
    if isinstance(Y_train, np.ndarray) or isinstance(Y_train, pd.DataFrame
        ) or isinstance(Y_train, pd.Series):
        shape_size = Y_train.shape
    elif isinstance(Y_train, list):
        shape_size = len(Y_train)
    else:
        shape_size = 'any'
    check_type = type(Y_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('Y_train')
        writer = csv.writer(f)
        writer.writerow(['Y_train', 109, check_type, shape_size])
if 'X_train' not in TANGSHAN:
    import csv
    if isinstance(X_train, np.ndarray) or isinstance(X_train, pd.DataFrame
        ) or isinstance(X_train, pd.Series):
        shape_size = X_train.shape
    elif isinstance(X_train, list):
        shape_size = len(X_train)
    else:
        shape_size = 'any'
    check_type = type(X_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X_train')
        writer = csv.writer(f)
        writer.writerow(['X_train', 109, check_type, shape_size])
X_test_xgb = xgb.DMatrix(X_test)
if 'X_test' not in TANGSHAN:
    import csv
    if isinstance(X_test, np.ndarray) or isinstance(X_test, pd.DataFrame
        ) or isinstance(X_test, pd.Series):
        shape_size = X_test.shape
    elif isinstance(X_test, list):
        shape_size = len(X_test)
    else:
        shape_size = 'any'
    check_type = type(X_test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X_test')
        writer = csv.writer(f)
        writer.writerow(['X_test', 111, check_type, shape_size])
gbm = xgb.train(params, T_train_xgb, 30)
Y_pred = gbm.predict(X_test_xgb)
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
        writer.writerow(['gbm', 117, check_type, shape_size])
if 'Y_pred' not in TANGSHAN:
    import csv
    if isinstance(Y_pred, np.ndarray) or isinstance(Y_pred, pd.DataFrame
        ) or isinstance(Y_pred, pd.Series):
        shape_size = Y_pred.shape
    elif isinstance(Y_pred, list):
        shape_size = len(Y_pred)
    else:
        shape_size = 'any'
    check_type = type(Y_pred)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('Y_pred')
        writer = csv.writer(f)
        writer.writerow(['Y_pred', 117, check_type, shape_size])
if 'X_test_xgb' not in TANGSHAN:
    import csv
    if isinstance(X_test_xgb, np.ndarray) or isinstance(X_test_xgb, pd.
        DataFrame) or isinstance(X_test_xgb, pd.Series):
        shape_size = X_test_xgb.shape
    elif isinstance(X_test_xgb, list):
        shape_size = len(X_test_xgb)
    else:
        shape_size = 'any'
    check_type = type(X_test_xgb)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X_test_xgb')
        writer = csv.writer(f)
        writer.writerow(['X_test_xgb', 117, check_type, shape_size])
submission = pd.DataFrame()
submission['QuoteNumber'] = test_df['QuoteNumber']
submission['QuoteConversion_Flag'] = Y_pred
if 'submission' not in TANGSHAN:
    import csv
    if isinstance(submission, np.ndarray) or isinstance(submission, pd.
        DataFrame) or isinstance(submission, pd.Series):
        shape_size = submission.shape
    elif isinstance(submission, list):
        shape_size = len(submission)
    else:
        shape_size = 'any'
    check_type = type(submission)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('submission')
        writer = csv.writer(f)
        writer.writerow(['submission', 125, check_type, shape_size])
submission.to_csv('homesite.csv', index=False)
