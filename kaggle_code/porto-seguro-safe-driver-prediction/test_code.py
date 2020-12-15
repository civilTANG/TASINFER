import numpy as np
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
        writer.writerow(['list', 11, check_type, shape_size])
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import datetime as dt
import gc
OPTIMIZE_ROUNDS = False
if 'OPTIMIZE_ROUNDS' not in TANGSHAN:
    import csv
    if isinstance(OPTIMIZE_ROUNDS, np.ndarray) or isinstance(OPTIMIZE_ROUNDS,
        pd.DataFrame) or isinstance(OPTIMIZE_ROUNDS, pd.Series):
        shape_size = OPTIMIZE_ROUNDS.shape
    elif isinstance(OPTIMIZE_ROUNDS, list):
        shape_size = len(OPTIMIZE_ROUNDS)
    else:
        shape_size = 'any'
    check_type = type(OPTIMIZE_ROUNDS)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('OPTIMIZE_ROUNDS')
        writer = csv.writer(f)
        writer.writerow(['OPTIMIZE_ROUNDS', 33, check_type, shape_size])


@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, val_series=None, tst_series=None, target
    =None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior

    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    averages = temp.groupby(by=trn_series.name)[target.name].agg(['mean',
        'count'])
    smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) /
        smoothing))
    prior = target.mean()
    averages[target.name] = prior * (1 - smoothing) + averages['mean'
        ] * smoothing
    averages.drop(['mean', 'count'], axis=1, inplace=True)
    ft_trn_series = pd.merge(trn_series.to_frame(trn_series.name), averages
        .reset_index().rename(columns={'index': target.name, target.name:
        'average'}), on=trn_series.name, how='left')['average'].rename(
        trn_series.name + '_mean').fillna(prior)
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(val_series.to_frame(val_series.name), averages
        .reset_index().rename(columns={'index': target.name, target.name:
        'average'}), on=val_series.name, how='left')['average'].rename(
        trn_series.name + '_mean').fillna(prior)
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(tst_series.to_frame(tst_series.name), averages
        .reset_index().rename(columns={'index': target.name, target.name:
        'average'}), on=tst_series.name, how='left')['average'].rename(
        trn_series.name + '_mean').fillna(prior)
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series,
        noise_level), add_noise(ft_tst_series, noise_level)


start_time = dt.datetime.now()
print('Started at ', start_time)
train_df = pd.read_csv('../input/train.csv', na_values='-1')
test_df = pd.read_csv('../input/test.csv', na_values='-1')
print('Finished data loading at ', dt.datetime.now())
train_features = ['ps_car_13', 'ps_ind_05_cat', 'ps_ind_03', 'ps_ind_15',
    'ps_reg_02', 'ps_car_14', 'ps_car_12', 'ps_car_01_cat', 'ps_car_07_cat',
    'ps_ind_17_bin', 'ps_reg_01', 'ps_ind_01', 'ps_ind_16_bin',
    'ps_ind_07_bin', 'ps_car_06_cat', 'ps_car_04_cat', 'ps_ind_06_bin',
    'ps_car_09_cat', 'ps_car_02_cat', 'ps_ind_02_cat', 'ps_car_11',
    'ps_car_05_cat', 'ps_calc_09', 'ps_calc_05', 'ps_ind_08_bin',
    'ps_car_08_cat', 'ps_ind_09_bin', 'ps_ind_04_cat', 'ps_ind_18_bin',
    'ps_ind_12_bin', 'ps_ind_14']
if 'train_features' not in TANGSHAN:
    import csv
    if isinstance(train_features, np.ndarray) or isinstance(train_features,
        pd.DataFrame) or isinstance(train_features, pd.Series):
        shape_size = train_features.shape
    elif isinstance(train_features, list):
        shape_size = len(train_features)
    else:
        shape_size = 'any'
    check_type = type(train_features)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_features')
        writer = csv.writer(f)
        writer.writerow(['train_features', 243, check_type, shape_size])
combs = [('ps_reg_01', 'ps_car_02_cat'), ('ps_reg_01', 'ps_car_04_cat')]
id_test = test_df['id'].values
if 'id_test' not in TANGSHAN:
    import csv
    if isinstance(id_test, np.ndarray) or isinstance(id_test, pd.DataFrame
        ) or isinstance(id_test, pd.Series):
        shape_size = id_test.shape
    elif isinstance(id_test, list):
        shape_size = len(id_test)
    else:
        shape_size = 'any'
    check_type = type(id_test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('id_test')
        writer = csv.writer(f)
        writer.writerow(['id_test', 325, check_type, shape_size])
id_train = train_df['id'].values
if 'id_train' not in TANGSHAN:
    import csv
    if isinstance(id_train, np.ndarray) or isinstance(id_train, pd.DataFrame
        ) or isinstance(id_train, pd.Series):
        shape_size = id_train.shape
    elif isinstance(id_train, list):
        shape_size = len(id_train)
    else:
        shape_size = 'any'
    check_type = type(id_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('id_train')
        writer = csv.writer(f)
        writer.writerow(['id_train', 327, check_type, shape_size])
y = train_df['target']
start = time.time()
if 'start' not in TANGSHAN:
    import csv
    if isinstance(start, np.ndarray) or isinstance(start, pd.DataFrame
        ) or isinstance(start, pd.Series):
        shape_size = start.shape
    elif isinstance(start, list):
        shape_size = len(start)
    else:
        shape_size = 'any'
    check_type = type(start)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('start')
        writer = csv.writer(f)
        writer.writerow(['start', 333, check_type, shape_size])
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + '_plus_' + f2
    print('current feature %60s %4d in %5.1f' % (name1, 
    if 'n_c' not in TANGSHAN:
        import csv
        if isinstance(n_c, np.ndarray) or isinstance(n_c, pd.DataFrame
            ) or isinstance(n_c, pd.Series):
            shape_size = n_c.shape
        elif isinstance(n_c, list):
            shape_size = len(n_c)
        else:
            shape_size = 'any'
        check_type = type(n_c)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('n_c')
            writer = csv.writer(f)
            writer.writerow(['n_c', 341, check_type, shape_size]), n_c + 1, (
                time.time() - start) / 60), end='')
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
            writer.writerow(['print', 339, check_type, shape_size])
    print('\r' * 75, end='')
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + '_' + train_df[f2
        ].apply(lambda x: str(x))
    if 'name1' not in TANGSHAN:
        import csv
        if isinstance(name1, np.ndarray) or isinstance(name1, pd.DataFrame
            ) or isinstance(name1, pd.Series):
            shape_size = name1.shape
        elif isinstance(name1, list):
            shape_size = len(name1)
        else:
            shape_size = 'any'
        check_type = type(name1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('name1')
            writer = csv.writer(f)
            writer.writerow(['name1', 345, check_type, shape_size])
    if 'train_df' not in TANGSHAN:
        import csv
        if isinstance(train_df, np.ndarray) or isinstance(train_df, pd.
            DataFrame) or isinstance(train_df, pd.Series):
            shape_size = train_df.shape
        elif isinstance(train_df, list):
            shape_size = len(train_df)
        else:
            shape_size = 'any'
        check_type = type(train_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train_df')
            writer = csv.writer(f)
            writer.writerow(['train_df', 345, check_type, shape_size])
    if 'f1' not in TANGSHAN:
        import csv
        if isinstance(f1, np.ndarray) or isinstance(f1, pd.DataFrame
            ) or isinstance(f1, pd.Series):
            shape_size = f1.shape
        elif isinstance(f1, list):
            shape_size = len(f1)
        else:
            shape_size = 'any'
        check_type = type(f1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('f1')
            writer = csv.writer(f)
            writer.writerow(['f1', 345, check_type, shape_size])
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + '_' + test_df[f2
        ].apply(lambda x: str(x))
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
            writer.writerow(['str', 347, check_type, shape_size])
    if 'x' not in TANGSHAN:
        import csv
        if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame
            ) or isinstance(x, pd.Series):
            shape_size = x.shape
        elif isinstance(x, list):
            shape_size = len(x)
        else:
            shape_size = 'any'
        check_type = type(x)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('x')
            writer = csv.writer(f)
            writer.writerow(['x', 347, check_type, shape_size])
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
            writer.writerow(['test_df', 347, check_type, shape_size])
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
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
            writer.writerow(['lbl', 353, check_type, shape_size])
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))
    train_features.append(name1)
if 'combs' not in TANGSHAN:
    import csv
    if isinstance(combs, np.ndarray) or isinstance(combs, pd.DataFrame
        ) or isinstance(combs, pd.Series):
        shape_size = combs.shape
    elif isinstance(combs, list):
        shape_size = len(combs)
    else:
        shape_size = 'any'
    check_type = type(combs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('combs')
        writer = csv.writer(f)
        writer.writerow(['combs', 335, check_type, shape_size])
if 'f2' not in TANGSHAN:
    import csv
    if isinstance(f2, np.ndarray) or isinstance(f2, pd.DataFrame
        ) or isinstance(f2, pd.Series):
        shape_size = f2.shape
    elif isinstance(f2, list):
        shape_size = len(f2)
    else:
        shape_size = 'any'
    check_type = type(f2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('f2')
        writer = csv.writer(f)
        writer.writerow(['f2', 335, check_type, shape_size])
X = train_df[train_features]
test_df = test_df[train_features]
X = X.fillna(X.mean())
if 'X' not in TANGSHAN:
    import csv
    if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or isinstance(X
        , pd.Series):
        shape_size = X.shape
    elif isinstance(X, list):
        shape_size = len(X)
    else:
        shape_size = 'any'
    check_type = type(X)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X')
        writer = csv.writer(f)
        writer.writerow(['X', 373, check_type, shape_size])
test_df = test_df.fillna(test_df.mean())
print('Training set details:')
print(X.info())
print('Testing set details:')
print(test_df.info())
f_cats = [f for f in X.columns if '_cat' in f]
y_valid_pred = 0 * y
if 'y' not in TANGSHAN:
    import csv
    if isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame) or isinstance(y
        , pd.Series):
        shape_size = y.shape
    elif isinstance(y, list):
        shape_size = len(y)
    else:
        shape_size = 'any'
    check_type = type(y)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y')
        writer = csv.writer(f)
        writer.writerow(['y', 391, check_type, shape_size])
y_test_pred = 0
print('Finished data pre-processing at ', dt.datetime.now())
K = 5
kf = KFold(n_splits=K, random_state=1, shuffle=True)
if 'K' not in TANGSHAN:
    import csv
    if isinstance(K, np.ndarray) or isinstance(K, pd.DataFrame) or isinstance(K
        , pd.Series):
        shape_size = K.shape
    elif isinstance(K, list):
        shape_size = len(K)
    else:
        shape_size = 'any'
    check_type = type(K)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('K')
        writer = csv.writer(f)
        writer.writerow(['K', 405, check_type, shape_size])
np.random.seed(0)
model = GradientBoostingClassifier(learning_rate=0.05, min_samples_split=
    5000, min_samples_leaf=40, max_depth=7, max_features='sqrt', subsample=
    0.9, random_state=10, n_estimators=190)
print('Finished setting up CV folds and classifier at ', dt.datetime.now())
print('Started CV at ', dt.datetime.now())
for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    if 'train_index' not in TANGSHAN:
        import csv
        if isinstance(train_index, np.ndarray) or isinstance(train_index,
            pd.DataFrame) or isinstance(train_index, pd.Series):
            shape_size = train_index.shape
        elif isinstance(train_index, list):
            shape_size = len(train_index)
        else:
            shape_size = 'any'
        check_type = type(train_index)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train_index')
            writer = csv.writer(f)
            writer.writerow(['train_index', 447, check_type, shape_size])
    if 'y_valid' not in TANGSHAN:
        import csv
        if isinstance(y_valid, np.ndarray) or isinstance(y_valid, pd.DataFrame
            ) or isinstance(y_valid, pd.Series):
            shape_size = y_valid.shape
        elif isinstance(y_valid, list):
            shape_size = len(y_valid)
        else:
            shape_size = 'any'
        check_type = type(y_valid)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y_valid')
            writer = csv.writer(f)
            writer.writerow(['y_valid', 447, check_type, shape_size])
    X_train, X_valid = X.iloc[(train_index), :].copy(), X.iloc[(test_index), :
        ].copy()
    X_test = test_df.copy()
    print('\nFold ', i)
    if 'i' not in TANGSHAN:
        import csv
        if isinstance(i, np.ndarray) or isinstance(i, pd.DataFrame
            ) or isinstance(i, pd.Series):
            shape_size = i.shape
        elif isinstance(i, list):
            shape_size = len(i)
        else:
            shape_size = 'any'
        check_type = type(i)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('i')
            writer = csv.writer(f)
            writer.writerow(['i', 453, check_type, shape_size])
    for f in f_cats:
        X_train[f + '_avg'], X_valid[f + '_avg'], X_test[f + '_avg'
            ] = target_encode(trn_series=X_train[f], val_series=X_valid[f],
            tst_series=X_test[f], target=y_train, min_samples_leaf=200,
            smoothing=10, noise_level=0)
    if 'f_cats' not in TANGSHAN:
        import csv
        if isinstance(f_cats, np.ndarray) or isinstance(f_cats, pd.DataFrame
            ) or isinstance(f_cats, pd.Series):
            shape_size = f_cats.shape
        elif isinstance(f_cats, list):
            shape_size = len(f_cats)
        else:
            shape_size = 'any'
        check_type = type(f_cats)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('f_cats')
            writer = csv.writer(f)
            writer.writerow(['f_cats', 459, check_type, shape_size])
    if OPTIMIZE_ROUNDS:
        eval_set = [(X_valid, y_valid)]
        if 'eval_set' not in TANGSHAN:
            import csv
            if isinstance(eval_set, np.ndarray) or isinstance(eval_set, pd.
                DataFrame) or isinstance(eval_set, pd.Series):
                shape_size = eval_set.shape
            elif isinstance(eval_set, list):
                shape_size = len(eval_set)
            else:
                shape_size = 'any'
            check_type = type(eval_set)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('eval_set')
                writer = csv.writer(f)
                writer.writerow(['eval_set', 483, check_type, shape_size])
        fit_model = model.fit(X_train, y_train)
    else:
        fit_model = model.fit(X_train, y_train)
        if 'fit_model' not in TANGSHAN:
            import csv
            if isinstance(fit_model, np.ndarray) or isinstance(fit_model,
                pd.DataFrame) or isinstance(fit_model, pd.Series):
                shape_size = fit_model.shape
            elif isinstance(fit_model, list):
                shape_size = len(fit_model)
            else:
                shape_size = 'any'
            check_type = type(fit_model)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('fit_model')
                writer = csv.writer(f)
                writer.writerow(['fit_model', 489, check_type, shape_size])
        if 'y_train' not in TANGSHAN:
            import csv
            if isinstance(y_train, np.ndarray) or isinstance(y_train, pd.
                DataFrame) or isinstance(y_train, pd.Series):
                shape_size = y_train.shape
            elif isinstance(y_train, list):
                shape_size = len(y_train)
            else:
                shape_size = 'any'
            check_type = type(y_train)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('y_train')
                writer = csv.writer(f)
                writer.writerow(['y_train', 489, check_type, shape_size])
    pred = model.predict_proba(X_valid)[:, (1)]
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
            writer.writerow(['model', 495, check_type, shape_size])
    if 'pred' not in TANGSHAN:
        import csv
        if isinstance(pred, np.ndarray) or isinstance(pred, pd.DataFrame
            ) or isinstance(pred, pd.Series):
            shape_size = pred.shape
        elif isinstance(pred, list):
            shape_size = len(pred)
        else:
            shape_size = 'any'
        check_type = type(pred)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pred')
            writer = csv.writer(f)
            writer.writerow(['pred', 495, check_type, shape_size])
    print('  Gini = ', eval_gini(y_valid, pred))
    y_valid_pred.iloc[test_index] = pred
    if 'test_index' not in TANGSHAN:
        import csv
        if isinstance(test_index, np.ndarray) or isinstance(test_index, pd.
            DataFrame) or isinstance(test_index, pd.Series):
            shape_size = test_index.shape
        elif isinstance(test_index, list):
            shape_size = len(test_index)
        else:
            shape_size = 'any'
        check_type = type(test_index)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('test_index')
            writer = csv.writer(f)
            writer.writerow(['test_index', 499, check_type, shape_size])
    if 'y_valid_pred' not in TANGSHAN:
        import csv
        if isinstance(y_valid_pred, np.ndarray) or isinstance(y_valid_pred,
            pd.DataFrame) or isinstance(y_valid_pred, pd.Series):
            shape_size = y_valid_pred.shape
        elif isinstance(y_valid_pred, list):
            shape_size = len(y_valid_pred)
        else:
            shape_size = 'any'
        check_type = type(y_valid_pred)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y_valid_pred')
            writer = csv.writer(f)
            writer.writerow(['y_valid_pred', 499, check_type, shape_size])
    y_test_pred += model.predict_proba(X_test)[:, (1)]
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
            writer.writerow(['X_test', 505, check_type, shape_size])
    del X_test, X_train, X_valid, y_train
    if 'X_valid' not in TANGSHAN:
        import csv
        if isinstance(X_valid, np.ndarray) or isinstance(X_valid, pd.DataFrame
            ) or isinstance(X_valid, pd.Series):
            shape_size = X_valid.shape
        elif isinstance(X_valid, list):
            shape_size = len(X_valid)
        else:
            shape_size = 'any'
        check_type = type(X_valid)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('X_valid')
            writer = csv.writer(f)
            writer.writerow(['X_valid', 509, check_type, shape_size])
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
            writer.writerow(['X_train', 509, check_type, shape_size])
if 'kf' not in TANGSHAN:
    import csv
    if isinstance(kf, np.ndarray) or isinstance(kf, pd.DataFrame
        ) or isinstance(kf, pd.Series):
        shape_size = kf.shape
    elif isinstance(kf, list):
        shape_size = len(kf)
    else:
        shape_size = 'any'
    check_type = type(kf)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('kf')
        writer = csv.writer(f)
        writer.writerow(['kf', 441, check_type, shape_size])
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
        writer.writerow(['enumerate', 441, check_type, shape_size])
print('Finished CV at ', dt.datetime.now())
y_test_pred /= K
if 'y_test_pred' not in TANGSHAN:
    import csv
    if isinstance(y_test_pred, np.ndarray) or isinstance(y_test_pred, pd.
        DataFrame) or isinstance(y_test_pred, pd.Series):
        shape_size = y_test_pred.shape
    elif isinstance(y_test_pred, list):
        shape_size = len(y_test_pred)
    else:
        shape_size = 'any'
    check_type = type(y_test_pred)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_test_pred')
        writer = csv.writer(f)
        writer.writerow(['y_test_pred', 517, check_type, shape_size])
print('Finished average test set predictions at ', dt.datetime.now())
sub = pd.DataFrame()
sub['id'] = id_test
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
        writer.writerow(['sub', 527, check_type, shape_size])
sub['target'] = y_test_pred
sub.to_csv('gbm_submit.csv', float_format='%.6f', index=False)
end_time = dt.datetime.now()
if 'end_time' not in TANGSHAN:
    import csv
    if isinstance(end_time, np.ndarray) or isinstance(end_time, pd.DataFrame
        ) or isinstance(end_time, pd.Series):
        shape_size = end_time.shape
    elif isinstance(end_time, list):
        shape_size = len(end_time)
    else:
        shape_size = 'any'
    check_type = type(end_time)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('end_time')
        writer = csv.writer(f)
        writer.writerow(['end_time', 535, check_type, shape_size])
elapsed_time = end_time - start_time
if 'elapsed_time' not in TANGSHAN:
    import csv
    if isinstance(elapsed_time, np.ndarray) or isinstance(elapsed_time, pd.
        DataFrame) or isinstance(elapsed_time, pd.Series):
        shape_size = elapsed_time.shape
    elif isinstance(elapsed_time, list):
        shape_size = len(elapsed_time)
    else:
        shape_size = 'any'
    check_type = type(elapsed_time)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('elapsed_time')
        writer = csv.writer(f)
        writer.writerow(['elapsed_time', 537, check_type, shape_size])
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
        writer.writerow(['start_time', 537, check_type, shape_size])
print('Finished the advanced gbm script ... ')
print('Elapsed time: ', elapsed_time)
