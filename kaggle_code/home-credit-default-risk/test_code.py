import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
test_2_df = pd.read_csv(
    '../input/home-credit-default-risk/application_test.csv')
test_1_df = pd.read_csv(
    '../input/home-credit-default-risk/application_train.csv')
train_df = pd.read_csv(
    '../input/home-credit-default-risk/previous_application.csv')
train_df = train_df.dropna(axis=0, subset=['CNT_PAYMENT'])
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
        writer.writerow(['train_df', 14, check_type, shape_size])
train_df['CNT_PAYMENT'] = train_df['CNT_PAYMENT'].astype('int')
for i in [test_1_df, test_2_df, train_df]:
    i['screwratio1'] = (i.AMT_CREDIT - i.AMT_GOODS_PRICE) / i.AMT_GOODS_PRICE
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
            writer.writerow(['i', 18, check_type, shape_size])
    i['screwratio2'] = (i.AMT_CREDIT - i.AMT_GOODS_PRICE) / i.AMT_CREDIT
    i['saint_CNT'] = i.AMT_CREDIT / i.AMT_ANNUITY
    i['angel_CNT'] = i.AMT_GOODS_PRICE / i.AMT_ANNUITY
    i['simple_diff'] = i.AMT_CREDIT - i.AMT_GOODS_PRICE
feats = ['saint_CNT', 'AMT_ANNUITY', 'angel_CNT', 'AMT_GOODS_PRICE',
    'screwratio2', 'screwratio1', 'AMT_CREDIT', 'simple_diff']
train_df = train_df.fillna(-1)
clf = LGBMClassifier(nthread=4, objective='multiclass', n_estimators=1000,
    learning_rate=0.02, num_leaves=50, max_depth=11, min_split_gain=
    0.0222415, min_child_weight=39.3259775, silent=-1, verbose=100)
print('fitting')
clf.fit(train_df[feats], train_df['CNT_PAYMENT'], verbose=500)
if 'feats' not in TANGSHAN:
    import csv
    if isinstance(feats, np.ndarray) or isinstance(feats, pd.DataFrame
        ) or isinstance(feats, pd.Series):
        shape_size = feats.shape
    elif isinstance(feats, list):
        shape_size = len(feats)
    else:
        shape_size = 'any'
    check_type = type(feats)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('feats')
        writer = csv.writer(f)
        writer.writerow(['feats', 39, check_type, shape_size])
print('training on previous apps done')
for frame in [[test_1_df, 'train'], [test_2_df, 'test']]:
    test_df = frame[0]
    if 'frame' not in TANGSHAN:
        import csv
        if isinstance(frame, np.ndarray) or isinstance(frame, pd.DataFrame
            ) or isinstance(frame, pd.Series):
            shape_size = frame.shape
        elif isinstance(frame, list):
            shape_size = len(frame)
        else:
            shape_size = 'any'
        check_type = type(frame)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('frame')
            writer = csv.writer(f)
            writer.writerow(['frame', 42, check_type, shape_size])
    tag = frame[1]
    if 'tag' not in TANGSHAN:
        import csv
        if isinstance(tag, np.ndarray) or isinstance(tag, pd.DataFrame
            ) or isinstance(tag, pd.Series):
            shape_size = tag.shape
        elif isinstance(tag, list):
            shape_size = len(tag)
        else:
            shape_size = 'any'
        check_type = type(tag)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('tag')
            writer = csv.writer(f)
            writer.writerow(['tag', 43, check_type, shape_size])
    j = clf.predict_proba(test_df[feats], verbose=500)
    if 'clf' not in TANGSHAN:
        import csv
        if isinstance(clf, np.ndarray) or isinstance(clf, pd.DataFrame
            ) or isinstance(clf, pd.Series):
            shape_size = clf.shape
        elif isinstance(clf, list):
            shape_size = len(clf)
        else:
            shape_size = 'any'
        check_type = type(clf)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('clf')
            writer = csv.writer(f)
            writer.writerow(['clf', 44, check_type, shape_size])
    test_df = test_df.fillna(-1)
    gc.collect()
    feature_importance_df = pd.DataFrame()
    sqsum = []
    if 'sqsum' not in TANGSHAN:
        import csv
        if isinstance(sqsum, np.ndarray) or isinstance(sqsum, pd.DataFrame
            ) or isinstance(sqsum, pd.Series):
            shape_size = sqsum.shape
        elif isinstance(sqsum, list):
            shape_size = len(sqsum)
        else:
            shape_size = 'any'
        check_type = type(sqsum)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('sqsum')
            writer = csv.writer(f)
            writer.writerow(['sqsum', 48, check_type, shape_size])
    test_df['certainty'] = 0
    print(np.arange(0, j.shape[1] - 1))
    print(j.shape)
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
            writer.writerow(['print', 51, check_type, shape_size])
    for k in np.arange(0, j.shape[1] - 1):
        test_df['CNT_prob_' + str(k)] = j[:, (k)]
        if 'k' not in TANGSHAN:
            import csv
            if isinstance(k, np.ndarray) or isinstance(k, pd.DataFrame
                ) or isinstance(k, pd.Series):
                shape_size = k.shape
            elif isinstance(k, list):
                shape_size = len(k)
            else:
                shape_size = 'any'
            check_type = type(k)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('k')
                writer = csv.writer(f)
                writer.writerow(['k', 53, check_type, shape_size])
        test_df['CNT_prob_sq_' + str(k)] = test_df['CNT_prob_' + str(k)
            ] * test_df['CNT_prob_' + str(k)]
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
                writer.writerow(['str', 54, check_type, shape_size])
        test_df['certainty'] += test_df['CNT_prob_sq_' + str(k)]
    if 'j' not in TANGSHAN:
        import csv
        if isinstance(j, np.ndarray) or isinstance(j, pd.DataFrame
            ) or isinstance(j, pd.Series):
            shape_size = j.shape
        elif isinstance(j, list):
            shape_size = len(j)
        else:
            shape_size = 'any'
        check_type = type(j)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('j')
            writer = csv.writer(f)
            writer.writerow(['j', 52, check_type, shape_size])
    predictions = pd.DataFrame()
    for k in np.arange(0, j.shape[1] - 1):
        predictions[str(clf.classes_[k])] = j[:, (k)]
        if 'predictions' not in TANGSHAN:
            import csv
            if isinstance(predictions, np.ndarray) or isinstance(predictions,
                pd.DataFrame) or isinstance(predictions, pd.Series):
                shape_size = predictions.shape
            elif isinstance(predictions, list):
                shape_size = len(predictions)
            else:
                shape_size = 'any'
            check_type = type(predictions)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('predictions')
                writer = csv.writer(f)
                writer.writerow(['predictions', 59, check_type, shape_size])
    predictions['best_guess'] = predictions.idxmax(axis=1)
    predictions['best_guess'] = predictions.best_guess.astype('int')
    test_df['lgbm_CNT'] = predictions['best_guess']
    print('starting the long, arduous task of computing interest rates')
    x = []
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
            writer.writerow(['x', 64, check_type, shape_size])
    for i in range(0, len(test_df.index)):
        x.append(np.rate(test_df['lgbm_CNT'][i], test_df['AMT_ANNUITY'][i],
            -test_df['AMT_CREDIT'][i], 0.0))
    test_df['rate_credit'] = x
    del x
    x = []
    for i in range(0, len(test_df.index)):
        x.append(np.rate(test_df['lgbm_CNT'][i], test_df['AMT_ANNUITY'][i],
            -test_df['AMT_GOODS_PRICE'][i], 0.0))
    if 'range' not in TANGSHAN:
        import csv
        if isinstance(range, np.ndarray) or isinstance(range, pd.DataFrame
            ) or isinstance(range, pd.Series):
            shape_size = range.shape
        elif isinstance(range, list):
            shape_size = len(range)
        else:
            shape_size = 'any'
        check_type = type(range)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('range')
            writer = csv.writer(f)
            writer.writerow(['range', 72, check_type, shape_size])
    test_df['rate_goods'] = x
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
            writer.writerow(['test_df', 74, check_type, shape_size])
    del x
    test_df[['rate_goods', 'SK_ID_CURR', 'lgbm_CNT', 'rate_credit',
        'certainty']].to_csv('lgbm_CNT_' + tag + '.csv', index=False)
if 'test_1_df' not in TANGSHAN:
    import csv
    if isinstance(test_1_df, np.ndarray) or isinstance(test_1_df, pd.DataFrame
        ) or isinstance(test_1_df, pd.Series):
        shape_size = test_1_df.shape
    elif isinstance(test_1_df, list):
        shape_size = len(test_1_df)
    else:
        shape_size = 'any'
    check_type = type(test_1_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test_1_df')
        writer = csv.writer(f)
        writer.writerow(['test_1_df', 41, check_type, shape_size])
if 'test_2_df' not in TANGSHAN:
    import csv
    if isinstance(test_2_df, np.ndarray) or isinstance(test_2_df, pd.DataFrame
        ) or isinstance(test_2_df, pd.Series):
        shape_size = test_2_df.shape
    elif isinstance(test_2_df, list):
        shape_size = len(test_2_df)
    else:
        shape_size = 'any'
    check_type = type(test_2_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test_2_df')
        writer = csv.writer(f)
        writer.writerow(['test_2_df', 41, check_type, shape_size])
feature_importance_df = pd.DataFrame()
if 'feature_importance_df' not in TANGSHAN:
    import csv
    if isinstance(feature_importance_df, np.ndarray) or isinstance(
        feature_importance_df, pd.DataFrame) or isinstance(
        feature_importance_df, pd.Series):
        shape_size = feature_importance_df.shape
    elif isinstance(feature_importance_df, list):
        shape_size = len(feature_importance_df)
    else:
        shape_size = 'any'
    check_type = type(feature_importance_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('feature_importance_df')
        writer = csv.writer(f)
        writer.writerow(['feature_importance_df', 77, check_type, shape_size])
feature_importance_df['feature'] = feats
feature_importance_df['importance'] = clf.feature_importances_
feature_importance_df[['feature', 'importance']].to_csv('importances.csv',
    index=False)


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[['feature', 'importance']].groupby('feature'
        ).mean().sort_values(by='importance', ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.
        feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x='importance', y='feature', data=best_features.sort_values
        (by='importance', ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


display_importances(feature_importance_df)
