import numpy as np
import pandas as pd
import os
print(os.listdir('../input'))
import time
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'), nrows=None)
if 'train_df' not in TANGSHAN:
    import csv
    if isinstance(train_df, np.ndarray) or isinstance(train_df, pd.DataFrame
        ) or isinstance(train_df, pd.Series):
        shape_size = train_df.shape
    elif isinstance(train_df, list):
        shape_size = len(train_df)
    else:
        shape_size = 'any'
    check_type = type(train_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_df')
        writer = csv.writer(f)
        writer.writerow(['train_df', 26, check_type, shape_size])
test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'), nrows=None)
if 'input_dir' not in TANGSHAN:
    import csv
    if isinstance(input_dir, np.ndarray) or isinstance(input_dir, pd.DataFrame
        ) or isinstance(input_dir, pd.Series):
        shape_size = input_dir.shape
    elif isinstance(input_dir, list):
        shape_size = len(input_dir)
    else:
        shape_size = 'any'
    check_type = type(input_dir)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('input_dir')
        writer = csv.writer(f)
        writer.writerow(['input_dir', 27, check_type, shape_size])
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
        writer.writerow(['test_df', 27, check_type, shape_size])
print('    Time elapsed %.0f sec' % (time.time() - start_time))
print('Using %d prediction variables' % (train_df.shape[1] - 2))
print('  Pre-processing data...')
train_df = train_df[train_df.parentesco1 == 1]
target = train_df.pop('Target')
len_train = len(train_df)
if 'len_train' not in TANGSHAN:
    import csv
    if isinstance(len_train, np.ndarray) or isinstance(len_train, pd.DataFrame
        ) or isinstance(len_train, pd.Series):
        shape_size = len_train.shape
    elif isinstance(len_train, list):
        shape_size = len(len_train)
    else:
        shape_size = 'any'
    check_type = type(len_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('len_train')
        writer = csv.writer(f)
        writer.writerow(['len_train', 43, check_type, shape_size])
merged_df = pd.concat([train_df, test_df])
print(merged_df.shape)
if 'merged_df' not in TANGSHAN:
    import csv
    if isinstance(merged_df, np.ndarray) or isinstance(merged_df, pd.DataFrame
        ) or isinstance(merged_df, pd.Series):
        shape_size = merged_df.shape
    elif isinstance(merged_df, list):
        shape_size = len(merged_df)
    else:
        shape_size = 'any'
    check_type = type(merged_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('merged_df')
        writer = csv.writer(f)
        writer.writerow(['merged_df', 45, check_type, shape_size])
del test_df, train_df
gc.collect()
print('  Time elapsed %.0f sec' % (time.time() - start_time))
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
        writer.writerow(['print', 48, check_type, shape_size])
merged_df.drop(['idhogar', 'SQBescolari', 'SQBage', 'SQBhogar_total',
    'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency',
    'SQBmeaned', 'agesq'], axis=1, inplace=True)
key = merged_df.pop('Id')
key = key[len_train:]
if 'key' not in TANGSHAN:
    import csv
    if isinstance(key, np.ndarray) or isinstance(key, pd.DataFrame
        ) or isinstance(key, pd.Series):
        shape_size = key.shape
    elif isinstance(key, list):
        shape_size = len(key)
    else:
        shape_size = 'any'
    check_type = type(key)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('key')
        writer = csv.writer(f)
        writer.writerow(['key', 56, check_type, shape_size])
merged_df.loc[merged_df.dependency == 'yes', 'dependency'] = 1
merged_df.loc[merged_df.dependency == 'no', 'dependency'] = 0
merged_df.loc[merged_df.edjefe == 'yes', 'edjefe'] = 1
merged_df.loc[merged_df.edjefe == 'no', 'edjefe'] = 0
merged_df.loc[merged_df.edjefa == 'yes', 'edjefa'] = 1
merged_df.loc[merged_df.edjefa == 'no', 'edjefa'] = 0
merged_df['dependency'] = merged_df['dependency'].astype(np.float32)
merged_df['edjefe'] = merged_df['edjefe'].astype(np.int8)
merged_df['edjefa'] = merged_df['edjefa'].astype(np.int8)
print(""" start training...
    Time elapsed %.0f sec""" % (time.time() -
    start_time))
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
        writer.writerow(['start_time', 75, check_type, shape_size])
params = {'max_depth': 8, 'task': 'train', 'boosting_type': 'gbdt',
    'objective': 'multiclass', 'num_class': 4, 'metric': '', 'num_leaves': 
    7, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 
    0.8, 'bagging_freq': 5, 'lambda_l1': 0, 'lambda_l2': 1, 'verbose': -1}
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
        writer.writerow(['params', 77, check_type, shape_size])
target = target - 1
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
        writer.writerow(['target', 95, check_type, shape_size])
num_folds = 5
if 'num_folds' not in TANGSHAN:
    import csv
    if isinstance(num_folds, np.ndarray) or isinstance(num_folds, pd.DataFrame
        ) or isinstance(num_folds, pd.Series):
        shape_size = num_folds.shape
    elif isinstance(num_folds, list):
        shape_size = len(num_folds)
    else:
        shape_size = 'any'
    check_type = type(num_folds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('num_folds')
        writer = csv.writer(f)
        writer.writerow(['num_folds', 96, check_type, shape_size])
test_x = merged_df[len_train:]
oof_preds = np.zeros([len_train])
sub_preds = np.zeros([test_x.shape[0], 4])
folds = KFold(n_splits=num_folds, shuffle=True, random_state=4564)
if 'folds' not in TANGSHAN:
    import csv
    if isinstance(folds, np.ndarray) or isinstance(folds, pd.DataFrame
        ) or isinstance(folds, pd.Series):
        shape_size = folds.shape
    elif isinstance(folds, list):
        shape_size = len(folds)
    else:
        shape_size = 'any'
    check_type = type(folds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('folds')
        writer = csv.writer(f)
        writer.writerow(['folds', 100, check_type, shape_size])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(merged_df[:
    len_train])):
    lgb_train = lgb.Dataset(merged_df.iloc[train_idx], target.iloc[train_idx])
    if 'train_idx' not in TANGSHAN:
        import csv
        if isinstance(train_idx, np.ndarray) or isinstance(train_idx, pd.
            DataFrame) or isinstance(train_idx, pd.Series):
            shape_size = train_idx.shape
        elif isinstance(train_idx, list):
            shape_size = len(train_idx)
        else:
            shape_size = 'any'
        check_type = type(train_idx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train_idx')
            writer = csv.writer(f)
            writer.writerow(['train_idx', 102, check_type, shape_size])
    lgb_valid = lgb.Dataset(merged_df.iloc[valid_idx], target.iloc[valid_idx])
    gbm = lgb.train(params, lgb_train, 5000, valid_sets=[lgb_train,
        lgb_valid], early_stopping_rounds=100, verbose_eval=1000)
    if 'lgb_train' not in TANGSHAN:
        import csv
        if isinstance(lgb_train, np.ndarray) or isinstance(lgb_train, pd.
            DataFrame) or isinstance(lgb_train, pd.Series):
            shape_size = lgb_train.shape
        elif isinstance(lgb_train, list):
            shape_size = len(lgb_train)
        else:
            shape_size = 'any'
        check_type = type(lgb_train)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('lgb_train')
            writer = csv.writer(f)
            writer.writerow(['lgb_train', 106, check_type, shape_size])
    if 'lgb_valid' not in TANGSHAN:
        import csv
        if isinstance(lgb_valid, np.ndarray) or isinstance(lgb_valid, pd.
            DataFrame) or isinstance(lgb_valid, pd.Series):
            shape_size = lgb_valid.shape
        elif isinstance(lgb_valid, list):
            shape_size = len(lgb_valid)
        else:
            shape_size = 'any'
        check_type = type(lgb_valid)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('lgb_valid')
            writer = csv.writer(f)
            writer.writerow(['lgb_valid', 106, check_type, shape_size])
    pr1 = gbm.predict(merged_df.iloc[valid_idx], num_iteration=gbm.
        best_iteration)
    if 'pr1' not in TANGSHAN:
        import csv
        if isinstance(pr1, np.ndarray) or isinstance(pr1, pd.DataFrame
            ) or isinstance(pr1, pd.Series):
            shape_size = pr1.shape
        elif isinstance(pr1, list):
            shape_size = len(pr1)
        else:
            shape_size = 'any'
        check_type = type(pr1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pr1')
            writer = csv.writer(f)
            writer.writerow(['pr1', 107, check_type, shape_size])
    pr2 = gbm.predict(test_x, num_iteration=gbm.best_iteration)
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
            writer.writerow(['gbm', 108, check_type, shape_size])
    if 'test_x' not in TANGSHAN:
        import csv
        if isinstance(test_x, np.ndarray) or isinstance(test_x, pd.DataFrame
            ) or isinstance(test_x, pd.Series):
            shape_size = test_x.shape
        elif isinstance(test_x, list):
            shape_size = len(test_x)
        else:
            shape_size = 'any'
        check_type = type(test_x)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('test_x')
            writer = csv.writer(f)
            writer.writerow(['test_x', 108, check_type, shape_size])
    pr1 = pr1 * np.array([1.2, 0.87, 1.135, 0.41])
    pr2 = pr2 * np.array([1.2, 0.87, 1.135, 0.41])
    if 'pr2' not in TANGSHAN:
        import csv
        if isinstance(pr2, np.ndarray) or isinstance(pr2, pd.DataFrame
            ) or isinstance(pr2, pd.Series):
            shape_size = pr2.shape
        elif isinstance(pr2, list):
            shape_size = len(pr2)
        else:
            shape_size = 'any'
        check_type = type(pr2)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pr2')
            writer = csv.writer(f)
            writer.writerow(['pr2', 110, check_type, shape_size])
    oof_preds[valid_idx] = pr1.argmax(axis=1)
    sub_preds += pr2 / folds.n_splits
    valid_idx += 1
if 'valid_idx' not in TANGSHAN:
    import csv
    if isinstance(valid_idx, np.ndarray) or isinstance(valid_idx, pd.DataFrame
        ) or isinstance(valid_idx, pd.Series):
        shape_size = valid_idx.shape
    elif isinstance(valid_idx, list):
        shape_size = len(valid_idx)
    else:
        shape_size = 'any'
    check_type = type(valid_idx)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('valid_idx')
        writer = csv.writer(f)
        writer.writerow(['valid_idx', 101, check_type, shape_size])
if 'enumerate' not in TANGSHAN:
    import csv
    if isinstance(enumerate, np.ndarray) or isinstance(enumerate, pd.DataFrame
        ) or isinstance(enumerate, pd.Series):
        shape_size = enumerate.shape
    elif isinstance(enumerate, list):
        shape_size = len(enumerate)
    else:
        shape_size = 'any'
    check_type = type(enumerate)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('enumerate')
        writer = csv.writer(f)
        writer.writerow(['enumerate', 101, check_type, shape_size])
if 'n_fold' not in TANGSHAN:
    import csv
    if isinstance(n_fold, np.ndarray) or isinstance(n_fold, pd.DataFrame
        ) or isinstance(n_fold, pd.Series):
        shape_size = n_fold.shape
    elif isinstance(n_fold, list):
        shape_size = len(n_fold)
    else:
        shape_size = 'any'
    check_type = type(n_fold)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('n_fold')
        writer = csv.writer(f)
        writer.writerow(['n_fold', 101, check_type, shape_size])
sub_preds = sub_preds.argmax(axis=1)
if 'sub_preds' not in TANGSHAN:
    import csv
    if isinstance(sub_preds, np.ndarray) or isinstance(sub_preds, pd.DataFrame
        ) or isinstance(sub_preds, pd.Series):
        shape_size = sub_preds.shape
    elif isinstance(sub_preds, list):
        shape_size = len(sub_preds)
    else:
        shape_size = 'any'
    check_type = type(sub_preds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('sub_preds')
        writer = csv.writer(f)
        writer.writerow(['sub_preds', 114, check_type, shape_size])
e = f1_score(target, oof_preds, average='macro')
if 'oof_preds' not in TANGSHAN:
    import csv
    if isinstance(oof_preds, np.ndarray) or isinstance(oof_preds, pd.DataFrame
        ) or isinstance(oof_preds, pd.Series):
        shape_size = oof_preds.shape
    elif isinstance(oof_preds, list):
        shape_size = len(oof_preds)
    else:
        shape_size = 'any'
    check_type = type(oof_preds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('oof_preds')
        writer = csv.writer(f)
        writer.writerow(['oof_preds', 115, check_type, shape_size])
print('Full validation score %.6f' % e)
if 'e' not in TANGSHAN:
    import csv
    if isinstance(e, np.ndarray) or isinstance(e, pd.DataFrame) or isinstance(e
        , pd.Series):
        shape_size = e.shape
    elif isinstance(e, list):
        shape_size = len(e)
    else:
        shape_size = 'any'
    check_type = type(e)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('e')
        writer = csv.writer(f)
        writer.writerow(['e', 116, check_type, shape_size])
print('    Time elapsed %.0f sec' % (time.time() - start_time))
out_df = pd.DataFrame({'Id': key})
out_df['Target'] = sub_preds.astype(np.float32)
out_df['Target'] = (out_df['Target'] + 0.5 + 1).astype(np.int8)
out_df['Target'] = np.maximum(out_df['Target'], 1)
if 'out_df' not in TANGSHAN:
    import csv
    if isinstance(out_df, np.ndarray) or isinstance(out_df, pd.DataFrame
        ) or isinstance(out_df, pd.Series):
        shape_size = out_df.shape
    elif isinstance(out_df, list):
        shape_size = len(out_df)
    else:
        shape_size = 'any'
    check_type = type(out_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('out_df')
        writer = csv.writer(f)
        writer.writerow(['out_df', 126, check_type, shape_size])
out_df['Target'] = np.minimum(out_df['Target'], 4)
out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.0f sec' % (time.time() - start_time))
