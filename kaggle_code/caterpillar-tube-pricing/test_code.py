"""
Caterpillar @ Kaggle
Adapted from arnaud demytt's R script
AND
Gilberto Titericz Junior's python scripts
__author__ = saihttam
"""
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
np.random.seed(42)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
if 'numerics' not in TANGSHAN:
    import csv
    if isinstance(numerics, np.ndarray) or isinstance(numerics, pd.DataFrame
        ) or isinstance(numerics, pd.Series):
        shape_size = numerics.shape
    elif isinstance(numerics, list):
        shape_size = len(numerics)
    else:
        shape_size = 'any'
    check_type = type(numerics)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('numerics')
        writer = csv.writer(f)
        writer.writerow(['numerics', 22, check_type, shape_size])
train = pd.read_csv(os.path.join('..', 'input', 'train_set.csv'),
    parse_dates=[2])
tube_data = pd.read_csv(os.path.join('..', 'input', 'tube.csv'))
train = pd.merge(train, tube_data, on='tube_assembly_id')
if 'tube_data' not in TANGSHAN:
    import csv
    if isinstance(tube_data, np.ndarray) or isinstance(tube_data, pd.DataFrame
        ) or isinstance(tube_data, pd.Series):
        shape_size = tube_data.shape
    elif isinstance(tube_data, list):
        shape_size = len(tube_data)
    else:
        shape_size = 'any'
    check_type = type(tube_data)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tube_data')
        writer = csv.writer(f)
        writer.writerow(['tube_data', 28, check_type, shape_size])
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['week'] = train.quote_date.dt.dayofyear % 52
train = train.drop(['quote_date', 'tube_assembly_id'], axis=1)
rs = ShuffleSplit(train.shape[0], n_iter=3, train_size=0.2, test_size=0.8,
    random_state=0)
for train_index, _ in rs:
    pass
if '_' not in TANGSHAN:
    import csv
    if isinstance(_, np.ndarray) or isinstance(_, pd.DataFrame) or isinstance(_
        , pd.Series):
        shape_size = _.shape
    elif isinstance(_, list):
        shape_size = len(_)
    else:
        shape_size = 'any'
    check_type = type(_)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('_')
        writer = csv.writer(f)
        writer.writerow(['_', 37, check_type, shape_size])
if 'rs' not in TANGSHAN:
    import csv
    if isinstance(rs, np.ndarray) or isinstance(rs, pd.DataFrame
        ) or isinstance(rs, pd.Series):
        shape_size = rs.shape
    elif isinstance(rs, list):
        shape_size = len(rs)
    else:
        shape_size = 'any'
    check_type = type(rs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('rs')
        writer = csv.writer(f)
        writer.writerow(['rs', 37, check_type, shape_size])
train = train.iloc[train_index]
if 'train_index' not in TANGSHAN:
    import csv
    if isinstance(train_index, np.ndarray) or isinstance(train_index, pd.
        DataFrame) or isinstance(train_index, pd.Series):
        shape_size = train_index.shape
    elif isinstance(train_index, list):
        shape_size = len(train_index)
    else:
        shape_size = 'any'
    check_type = type(train_index)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_index')
        writer = csv.writer(f)
        writer.writerow(['train_index', 40, check_type, shape_size])
print(train.shape)
newdf = train.select_dtypes(include=numerics)
if 'newdf' not in TANGSHAN:
    import csv
    if isinstance(newdf, np.ndarray) or isinstance(newdf, pd.DataFrame
        ) or isinstance(newdf, pd.Series):
        shape_size = newdf.shape
    elif isinstance(newdf, list):
        shape_size = len(newdf)
    else:
        shape_size = 'any'
    check_type = type(newdf)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('newdf')
        writer = csv.writer(f)
        writer.writerow(['newdf', 43, check_type, shape_size])
numcolumns = newdf.columns.values
allcolumns = train.columns.values
nonnumcolumns = list(set(allcolumns) - set(numcolumns))
if 'allcolumns' not in TANGSHAN:
    import csv
    if isinstance(allcolumns, np.ndarray) or isinstance(allcolumns, pd.
        DataFrame) or isinstance(allcolumns, pd.Series):
        shape_size = allcolumns.shape
    elif isinstance(allcolumns, list):
        shape_size = len(allcolumns)
    else:
        shape_size = 'any'
    check_type = type(allcolumns)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('allcolumns')
        writer = csv.writer(f)
        writer.writerow(['allcolumns', 47, check_type, shape_size])
if 'set' not in TANGSHAN:
    import csv
    if isinstance(set, np.ndarray) or isinstance(set, pd.DataFrame
        ) or isinstance(set, pd.Series):
        shape_size = set.shape
    elif isinstance(set, list):
        shape_size = len(set)
    else:
        shape_size = 'any'
    check_type = type(set)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('set')
        writer = csv.writer(f)
        writer.writerow(['set', 47, check_type, shape_size])
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
        writer.writerow(['list', 47, check_type, shape_size])
print('Numcolumns %s ' % numcolumns)
if 'numcolumns' not in TANGSHAN:
    import csv
    if isinstance(numcolumns, np.ndarray) or isinstance(numcolumns, pd.
        DataFrame) or isinstance(numcolumns, pd.Series):
        shape_size = numcolumns.shape
    elif isinstance(numcolumns, list):
        shape_size = len(numcolumns)
    else:
        shape_size = 'any'
    check_type = type(numcolumns)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('numcolumns')
        writer = csv.writer(f)
        writer.writerow(['numcolumns', 48, check_type, shape_size])
print('Nonnumcolumns %s ' % nonnumcolumns)
print("""Nans before processing: 
 {0}""".format(train.isnull().sum()))
train[numcolumns] = train[numcolumns].fillna(-999999)
train[nonnumcolumns] = train[nonnumcolumns].fillna('NAvalue')
print("""Nans after processing: 
 {0}""".format(train.isnull().sum()))
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
        writer.writerow(['print', 54, check_type, shape_size])
for col in nonnumcolumns:
    ser = train[col]
    counts = ser.value_counts().keys()
    if 'counts' not in TANGSHAN:
        import csv
        if isinstance(counts, np.ndarray) or isinstance(counts, pd.DataFrame
            ) or isinstance(counts, pd.Series):
            shape_size = counts.shape
        elif isinstance(counts, list):
            shape_size = len(counts)
        else:
            shape_size = 'any'
        check_type = type(counts)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('counts')
            writer = csv.writer(f)
            writer.writerow(['counts', 58, check_type, shape_size])
    if 'ser' not in TANGSHAN:
        import csv
        if isinstance(ser, np.ndarray) or isinstance(ser, pd.DataFrame
            ) or isinstance(ser, pd.Series):
            shape_size = ser.shape
        elif isinstance(ser, list):
            shape_size = len(ser)
        else:
            shape_size = 'any'
        check_type = type(ser)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('ser')
            writer = csv.writer(f)
            writer.writerow(['ser', 58, check_type, shape_size])
    threshold = 5
    if 'threshold' not in TANGSHAN:
        import csv
        if isinstance(threshold, np.ndarray) or isinstance(threshold, pd.
            DataFrame) or isinstance(threshold, pd.Series):
            shape_size = threshold.shape
        elif isinstance(threshold, list):
            shape_size = len(threshold)
        else:
            shape_size = 'any'
        check_type = type(threshold)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('threshold')
            writer = csv.writer(f)
            writer.writerow(['threshold', 60, check_type, shape_size])
    if len(counts) > threshold:
        ser[~ser.isin(counts[:threshold])] = 'rareValue'
    if len(counts) <= 1:
        print('Dropping Column %s with %d values' % (col, len(counts)))
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
                writer.writerow(['col', 64, check_type, shape_size])
        train = train.drop(col, axis=1)
    else:
        train[col] = ser.astype('category')
if 'nonnumcolumns' not in TANGSHAN:
    import csv
    if isinstance(nonnumcolumns, np.ndarray) or isinstance(nonnumcolumns,
        pd.DataFrame) or isinstance(nonnumcolumns, pd.Series):
        shape_size = nonnumcolumns.shape
    elif isinstance(nonnumcolumns, list):
        shape_size = len(nonnumcolumns)
    else:
        shape_size = 'any'
    check_type = type(nonnumcolumns)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('nonnumcolumns')
        writer = csv.writer(f)
        writer.writerow(['nonnumcolumns', 56, check_type, shape_size])
train = pd.get_dummies(train)
print('Size after dummies {0}'.format(train.shape))
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
        writer.writerow(['train', 70, check_type, shape_size])
train['logquantity'] = np.log(train['quantity'])
train['log1usage'] = np.log1p(train['annual_usage'])
train['log1radius'] = np.log1p(train['bend_radius'])
train['log1length'] = np.log1p(train['length'])
train = train.drop(['quantity', 'annual_usage', 'bend_radius', 'length'],
    axis=1)
labels = train.cost.values
if 'labels' not in TANGSHAN:
    import csv
    if isinstance(labels, np.ndarray) or isinstance(labels, pd.DataFrame
        ) or isinstance(labels, pd.Series):
        shape_size = labels.shape
    elif isinstance(labels, list):
        shape_size = len(labels)
    else:
        shape_size = 'any'
    check_type = type(labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('labels')
        writer = csv.writer(f)
        writer.writerow(['labels', 79, check_type, shape_size])
Xtrain = train.drop(['cost'], axis=1)
names = list(Xtrain.columns.values)
if 'names' not in TANGSHAN:
    import csv
    if isinstance(names, np.ndarray) or isinstance(names, pd.DataFrame
        ) or isinstance(names, pd.Series):
        shape_size = names.shape
    elif isinstance(names, list):
        shape_size = len(names)
    else:
        shape_size = 'any'
    check_type = type(names)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('names')
        writer = csv.writer(f)
        writer.writerow(['names', 81, check_type, shape_size])
Xtrain = np.array(Xtrain)
label_log = np.log1p(labels)
if 'label_log' not in TANGSHAN:
    import csv
    if isinstance(label_log, np.ndarray) or isinstance(label_log, pd.DataFrame
        ) or isinstance(label_log, pd.Series):
        shape_size = label_log.shape
    elif isinstance(label_log, list):
        shape_size = len(label_log)
    else:
        shape_size = 'any'
    check_type = type(label_log)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('label_log')
        writer = csv.writer(f)
        writer.writerow(['label_log', 84, check_type, shape_size])
Xtrain, label_log = shuffle(Xtrain, label_log, random_state=666)
if 'Xtrain' not in TANGSHAN:
    import csv
    if isinstance(Xtrain, np.ndarray) or isinstance(Xtrain, pd.DataFrame
        ) or isinstance(Xtrain, pd.Series):
        shape_size = Xtrain.shape
    elif isinstance(Xtrain, list):
        shape_size = len(Xtrain)
    else:
        shape_size = 'any'
    check_type = type(Xtrain)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('Xtrain')
        writer = csv.writer(f)
        writer.writerow(['Xtrain', 85, check_type, shape_size])
model = ExtraTreesClassifier(n_estimators=50, max_depth=15)
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
        writer.writerow(['model', 87, check_type, shape_size])
model.fit(Xtrain, label_log)
features = []
importances = model.feature_importances_
if 'importances' not in TANGSHAN:
    import csv
    if isinstance(importances, np.ndarray) or isinstance(importances, pd.
        DataFrame) or isinstance(importances, pd.Series):
        shape_size = importances.shape
    elif isinstance(importances, list):
        shape_size = len(importances)
    else:
        shape_size = 'any'
    check_type = type(importances)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('importances')
        writer = csv.writer(f)
        writer.writerow(['importances', 92, check_type, shape_size])
indices = np.argsort(importances)[::-1]
for f in range(len(importances)):
    print('%d. feature %d (%f), %s' % (f + 1, indices[f], importances[
        indices[f]], names[indices[f]]))
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
            writer.writerow(['indices', 96, check_type, shape_size])
    features.append(indices[f])
    if len(features) >= 5:
        break
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
        writer.writerow(['range', 95, check_type, shape_size])
if 'f' not in TANGSHAN:
    import csv
    if isinstance(f, np.ndarray) or isinstance(f, pd.DataFrame) or isinstance(f
        , pd.Series):
        shape_size = f.shape
    elif isinstance(f, list):
        shape_size = len(f)
    else:
        shape_size = 'any'
    check_type = type(f)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('f')
        writer = csv.writer(f)
        writer.writerow(['f', 95, check_type, shape_size])
q = pd.qcut(train['cost'], 5)
if 'q' not in TANGSHAN:
    import csv
    if isinstance(q, np.ndarray) or isinstance(q, pd.DataFrame) or isinstance(q
        , pd.Series):
        shape_size = q.shape
    elif isinstance(q, list):
        shape_size = len(q)
    else:
        shape_size = 'any'
    check_type = type(q)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('q')
        writer = csv.writer(f)
        writer.writerow(['q', 102, check_type, shape_size])
print('Bins are {0}'.format(q))
train['cost_5'] = q
fig = plt.figure()
featurenames = [names[feature] for feature in features]
if 'feature' not in TANGSHAN:
    import csv
    if isinstance(feature, np.ndarray) or isinstance(feature, pd.DataFrame
        ) or isinstance(feature, pd.Series):
        shape_size = feature.shape
    elif isinstance(feature, list):
        shape_size = len(feature)
    else:
        shape_size = 'any'
    check_type = type(feature)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('feature')
        writer = csv.writer(f)
        writer.writerow(['feature', 107, check_type, shape_size])
if 'featurenames' not in TANGSHAN:
    import csv
    if isinstance(featurenames, np.ndarray) or isinstance(featurenames, pd.
        DataFrame) or isinstance(featurenames, pd.Series):
        shape_size = featurenames.shape
    elif isinstance(featurenames, list):
        shape_size = len(featurenames)
    else:
        shape_size = 'any'
    check_type = type(featurenames)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('featurenames')
        writer = csv.writer(f)
        writer.writerow(['featurenames', 107, check_type, shape_size])
featurenames.append('cost_5')
pg = sns.pairplot(train[featurenames], hue='cost_5', size=2.5)
if 'pg' not in TANGSHAN:
    import csv
    if isinstance(pg, np.ndarray) or isinstance(pg, pd.DataFrame
        ) or isinstance(pg, pd.Series):
        shape_size = pg.shape
    elif isinstance(pg, list):
        shape_size = len(pg)
    else:
        shape_size = 'any'
    check_type = type(pg)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('pg')
        writer = csv.writer(f)
        writer.writerow(['pg', 109, check_type, shape_size])
pg.savefig('pairplotquintile.png')
print('Training GBRT...')
clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
    learning_rate=0.1, loss='huber', random_state=1)
clf.fit(Xtrain, label_log)
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
        writer.writerow(['clf', 117, check_type, shape_size])
print('Convenience plot with ``partial_dependence_plots``')
target_feature = features[0], features[1]
if 'target_feature' not in TANGSHAN:
    import csv
    if isinstance(target_feature, np.ndarray) or isinstance(target_feature,
        pd.DataFrame) or isinstance(target_feature, pd.Series):
        shape_size = target_feature.shape
    elif isinstance(target_feature, list):
        shape_size = len(target_feature)
    else:
        shape_size = 'any'
    check_type = type(target_feature)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('target_feature')
        writer = csv.writer(f)
        writer.writerow(['target_feature', 121, check_type, shape_size])
features.append(target_feature)
fig, axs = plot_partial_dependence(clf, Xtrain, features, feature_names=
    names, n_jobs=3, grid_resolution=50)
if 'axs' not in TANGSHAN:
    import csv
    if isinstance(axs, np.ndarray) or isinstance(axs, pd.DataFrame
        ) or isinstance(axs, pd.Series):
        shape_size = axs.shape
    elif isinstance(axs, list):
        shape_size = len(axs)
    else:
        shape_size = 'any'
    check_type = type(axs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('axs')
        writer = csv.writer(f)
        writer.writerow(['axs', 123, check_type, shape_size])
if 'features' not in TANGSHAN:
    import csv
    if isinstance(features, np.ndarray) or isinstance(features, pd.DataFrame
        ) or isinstance(features, pd.Series):
        shape_size = features.shape
    elif isinstance(features, list):
        shape_size = len(features)
    else:
        shape_size = 'any'
    check_type = type(features)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('features')
        writer = csv.writer(f)
        writer.writerow(['features', 123, check_type, shape_size])
if 'fig' not in TANGSHAN:
    import csv
    if isinstance(fig, np.ndarray) or isinstance(fig, pd.DataFrame
        ) or isinstance(fig, pd.Series):
        shape_size = fig.shape
    elif isinstance(fig, list):
        shape_size = len(fig)
    else:
        shape_size = 'any'
    check_type = type(fig)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('fig')
        writer = csv.writer(f)
        writer.writerow(['fig', 123, check_type, shape_size])
fig.savefig('partial.png')
