import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
TANGSHAN = []
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_train = np.log(train['loss'])
indices = test['id']
if 'indices' not in TANGSHAN:
    import csv
    if isinstance(indices, np.ndarray) or isinstance(indices, pd.DataFrame
        ) or isinstance(indices, pd.Series):
        shape_size = indices.shape
    elif isinstance(indices, list):
        shape_size = len(indices)
    else:
        shape_size = 'any'
    check_type = type(indices)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('indices')
        writer = csv.writer(f)
        writer.writerow(['indices', 16, check_type, shape_size])
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
        writer.writerow(['test', 16, check_type, shape_size])
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
            writer.writerow(['train', 22, check_type, shape_size])
train.drop(['id', 'loss'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
categorical = [col for col in train.columns if 'cat' in col]
if 'categorical' not in TANGSHAN:
    import csv
    if isinstance(categorical, np.ndarray) or isinstance(categorical, pd.
        DataFrame) or isinstance(categorical, pd.Series):
        shape_size = categorical.shape
    elif isinstance(categorical, list):
        shape_size = len(categorical)
    else:
        shape_size = 'any'
    check_type = type(categorical)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('categorical')
        writer = csv.writer(f)
        writer.writerow(['categorical', 21, check_type, shape_size])
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
            writer.writerow(['train', 22, check_type, shape_size])
continuous = [col for col in train.columns if 'cont' in col]
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
        writer.writerow(['train', 22, check_type, shape_size])
num_train_rows = train.shape[0]
if 'num_train_rows' not in TANGSHAN:
    import csv
    if isinstance(num_train_rows, np.ndarray) or isinstance(num_train_rows,
        pd.DataFrame) or isinstance(num_train_rows, pd.Series):
        shape_size = num_train_rows.shape
    elif isinstance(num_train_rows, list):
        shape_size = len(num_train_rows)
    else:
        shape_size = 'any'
    check_type = type(num_train_rows)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('num_train_rows')
        writer = csv.writer(f)
        writer.writerow(['num_train_rows', 24, check_type, shape_size])
df = pd.concat((train, test)).reset_index(drop=True)
mms = MinMaxScaler()
for col in continuous:
    shift = np.abs(np.floor(min(df[col])))
    if 'min' not in TANGSHAN:
        import csv
        if isinstance(min, np.ndarray) or isinstance(min, pd.DataFrame
            ) or isinstance(min, pd.Series):
            shape_size = min.shape
        elif isinstance(min, list):
            shape_size = len(min)
        else:
            shape_size = 'any'
        check_type = type(min)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('min')
            writer = csv.writer(f)
            writer.writerow(['min', 29, check_type, shape_size])
    if 'shift' not in TANGSHAN:
        import csv
        if isinstance(shift, np.ndarray) or isinstance(shift, pd.DataFrame
            ) or isinstance(shift, pd.Series):
            shape_size = shift.shape
        elif isinstance(shift, list):
            shape_size = len(shift)
        else:
            shape_size = 'any'
        check_type = type(shift)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('shift')
            writer = csv.writer(f)
            writer.writerow(['shift', 29, check_type, shape_size])
    df[col] = mms.fit_transform(stats.boxcox(df[col] + shift + 1)[0].
        reshape(-1, 1))
    if 'mms' not in TANGSHAN:
        import csv
        if isinstance(mms, np.ndarray) or isinstance(mms, pd.DataFrame
            ) or isinstance(mms, pd.Series):
            shape_size = mms.shape
        elif isinstance(mms, list):
            shape_size = len(mms)
        else:
            shape_size = 'any'
        check_type = type(mms)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('mms')
            writer = csv.writer(f)
            writer.writerow(['mms', 30, check_type, shape_size])
if 'continuous' not in TANGSHAN:
    import csv
    if isinstance(continuous, np.ndarray) or isinstance(continuous, pd.
        DataFrame) or isinstance(continuous, pd.Series):
        shape_size = continuous.shape
    elif isinstance(continuous, list):
        shape_size = len(continuous)
    else:
        shape_size = 'any'
    check_type = type(continuous)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('continuous')
        writer = csv.writer(f)
        writer.writerow(['continuous', 28, check_type, shape_size])
le = LabelEncoder()
for col in categorical:
    df[col] = le.fit_transform(df[col])
    if 'le' not in TANGSHAN:
        import csv
        if isinstance(le, np.ndarray) or isinstance(le, pd.DataFrame
            ) or isinstance(le, pd.Series):
            shape_size = le.shape
        elif isinstance(le, list):
            shape_size = len(le)
        else:
            shape_size = 'any'
        check_type = type(le)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('le')
            writer = csv.writer(f)
            writer.writerow(['le', 35, check_type, shape_size])
x_train = np.array(df.iloc[:num_train_rows, :])
x_test = np.array(df.iloc[num_train_rows:, :])
if 'df' not in TANGSHAN:
    import csv
    if isinstance(df, np.ndarray) or isinstance(df, pd.DataFrame
        ) or isinstance(df, pd.Series):
        shape_size = df.shape
    elif isinstance(df, list):
        shape_size = len(df)
    else:
        shape_size = 'any'
    check_type = type(df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('df')
        writer = csv.writer(f)
        writer.writerow(['df', 38, check_type, shape_size])
params = {}
params['booster'] = 'gbtree'
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['eta'] = 0.075
params['gamma'] = 0.529
params['min_child_weight'] = 4.2922
params['num_parallel_tree'] = 1
params['colsample_bytree'] = 0.3085
params['subsample'] = 0.993
params['max_depth'] = 7
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
        writer.writerow(['params', 50, check_type, shape_size])
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 1337
dtrain = xgb.DMatrix(x_train, label=y_train)
if 'x_train' not in TANGSHAN:
    import csv
    if isinstance(x_train, np.ndarray) or isinstance(x_train, pd.DataFrame
        ) or isinstance(x_train, pd.Series):
        shape_size = x_train.shape
    elif isinstance(x_train, list):
        shape_size = len(x_train)
    else:
        shape_size = 'any'
    check_type = type(x_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x_train')
        writer = csv.writer(f)
        writer.writerow(['x_train', 55, check_type, shape_size])
dtest = xgb.DMatrix(x_test)
if 'x_test' not in TANGSHAN:
    import csv
    if isinstance(x_test, np.ndarray) or isinstance(x_test, pd.DataFrame
        ) or isinstance(x_test, pd.Series):
        shape_size = x_test.shape
    elif isinstance(x_test, list):
        shape_size = len(x_test)
    else:
        shape_size = 'any'
    check_type = type(x_test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x_test')
        writer = csv.writer(f)
        writer.writerow(['x_test', 56, check_type, shape_size])


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


res = xgb.cv(params, dtrain, num_boost_round=100, nfold=4, seed=1337,
    stratified=False, early_stopping_rounds=15, verbose_eval=10, show_stdv=
    True, feval=xg_eval_mae, maximize=False)
best_nrounds = res.shape[0] - 1
if 'best_nrounds' not in TANGSHAN:
    import csv
    if isinstance(best_nrounds, np.ndarray) or isinstance(best_nrounds, pd.
        DataFrame) or isinstance(best_nrounds, pd.Series):
        shape_size = best_nrounds.shape
    elif isinstance(best_nrounds, list):
        shape_size = len(best_nrounds)
    else:
        shape_size = 'any'
    check_type = type(best_nrounds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('best_nrounds')
        writer = csv.writer(f)
        writer.writerow(['best_nrounds', 74, check_type, shape_size])
cv_mean = res.iloc[-1, 0]
if 'res' not in TANGSHAN:
    import csv
    if isinstance(res, np.ndarray) or isinstance(res, pd.DataFrame
        ) or isinstance(res, pd.Series):
        shape_size = res.shape
    elif isinstance(res, list):
        shape_size = len(res)
    else:
        shape_size = 'any'
    check_type = type(res)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('res')
        writer = csv.writer(f)
        writer.writerow(['res', 75, check_type, shape_size])
if 'cv_mean' not in TANGSHAN:
    import csv
    if isinstance(cv_mean, np.ndarray) or isinstance(cv_mean, pd.DataFrame
        ) or isinstance(cv_mean, pd.Series):
        shape_size = cv_mean.shape
    elif isinstance(cv_mean, list):
        shape_size = len(cv_mean)
    else:
        shape_size = 'any'
    check_type = type(cv_mean)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cv_mean')
        writer = csv.writer(f)
        writer.writerow(['cv_mean', 75, check_type, shape_size])
cv_std = res.iloc[-1, 1]
if 'cv_std' not in TANGSHAN:
    import csv
    if isinstance(cv_std, np.ndarray) or isinstance(cv_std, pd.DataFrame
        ) or isinstance(cv_std, pd.Series):
        shape_size = cv_std.shape
    elif isinstance(cv_std, list):
        shape_size = len(cv_std)
    else:
        shape_size = 'any'
    check_type = type(cv_std)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cv_std')
        writer = csv.writer(f)
        writer.writerow(['cv_std', 76, check_type, shape_size])
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))
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
        writer.writerow(['print', 77, check_type, shape_size])
model = xgb.train(params, dtrain, best_nrounds)
if 'dtrain' not in TANGSHAN:
    import csv
    if isinstance(dtrain, np.ndarray) or isinstance(dtrain, pd.DataFrame
        ) or isinstance(dtrain, pd.Series):
        shape_size = dtrain.shape
    elif isinstance(dtrain, list):
        shape_size = len(dtrain)
    else:
        shape_size = 'any'
    check_type = type(dtrain)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dtrain')
        writer = csv.writer(f)
        writer.writerow(['dtrain', 79, check_type, shape_size])
predictions = np.exp(model.predict(dtest))
if 'dtest' not in TANGSHAN:
    import csv
    if isinstance(dtest, np.ndarray) or isinstance(dtest, pd.DataFrame
        ) or isinstance(dtest, pd.Series):
        shape_size = dtest.shape
    elif isinstance(dtest, list):
        shape_size = len(dtest)
    else:
        shape_size = 'any'
    check_type = type(dtest)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dtest')
        writer = csv.writer(f)
        writer.writerow(['dtest', 81, check_type, shape_size])
if 'model' not in TANGSHAN:
    import csv
    if isinstance(model, np.ndarray) or isinstance(model, pd.DataFrame
        ) or isinstance(model, pd.Series):
        shape_size = model.shape
    elif isinstance(model, list):
        shape_size = len(model)
    else:
        shape_size = 'any'
    check_type = type(model)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('model')
        writer = csv.writer(f)
        writer.writerow(['model', 81, check_type, shape_size])
submission = pd.DataFrame({'id': indices, 'loss': predictions})
submission.to_csv('submission.csv', index=None)
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
        writer.writerow(['submission', 84, check_type, shape_size])
plt.xlim(-1000, 15000)
sns.kdeplot(np.exp(y_train))
if 'y_train' not in TANGSHAN:
    import csv
    if isinstance(y_train, np.ndarray) or isinstance(y_train, pd.DataFrame
        ) or isinstance(y_train, pd.Series):
        shape_size = y_train.shape
    elif isinstance(y_train, list):
        shape_size = len(y_train)
    else:
        shape_size = 'any'
    check_type = type(y_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_train')
        writer = csv.writer(f)
        writer.writerow(['y_train', 87, check_type, shape_size])
sns.kdeplot(predictions)
if 'predictions' not in TANGSHAN:
    import csv
    if isinstance(predictions, np.ndarray) or isinstance(predictions, pd.
        DataFrame) or isinstance(predictions, pd.Series):
        shape_size = predictions.shape
    elif isinstance(predictions, list):
        shape_size = len(predictions)
    else:
        shape_size = 'any'
    check_type = type(predictions)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('predictions')
        writer = csv.writer(f)
        writer.writerow(['predictions', 88, check_type, shape_size])
plt.savefig('submission.png')
