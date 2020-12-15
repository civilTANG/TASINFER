import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import re
import numpy as np
warnings.filterwarnings('ignore')
train = pd.read_csv(
    'https://github.com/dariyush/Data-Science/blob/master/train.csv')
test = pd.read_csv('../input/test.csv')
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
        if 'event_type' not in TANGSHAN:
            import csv
            if isinstance(event_type, np.ndarray) or isinstance(event_type,
                pd.DataFrame) or isinstance(event_type, pd.Series):
                shape_size = event_type.shape
            elif isinstance(event_type, list):
                shape_size = len(event_type)
            else:
                shape_size = 'any'
            check_type = type(event_type)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('event_type')
                writer = csv.writer(f)
                writer.writerow(['event_type', 25, check_type, shape_size])
        writer.writerow(['test', 24, check_type, shape_size])
event_type = pd.read_csv('../input/event_type.csv')
if 'event_type' not in TANGSHAN:
    import csv
    if isinstance(event_type, np.ndarray) or isinstance(event_type, pd.
        DataFrame) or isinstance(event_type, pd.Series):
        shape_size = event_type.shape
    elif isinstance(event_type, list):
        shape_size = len(event_type)
    else:
        shape_size = 'any'
    check_type = type(event_type)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('event_type')
        writer = csv.writer(f)
        writer.writerow(['event_type', 25, check_type, shape_size])
log_feature = pd.read_csv('../input/log_feature.csv')
if 'log_feature' not in TANGSHAN:
    import csv
    if isinstance(log_feature, np.ndarray) or isinstance(log_feature, pd.
        DataFrame) or isinstance(log_feature, pd.Series):
        shape_size = log_feature.shape
    elif isinstance(log_feature, list):
        shape_size = len(log_feature)
    else:
        shape_size = 'any'
    check_type = type(log_feature)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('log_feature')
        writer = csv.writer(f)
        writer.writerow(['log_feature', 26, check_type, shape_size])
resource_type = pd.read_csv('../input/resource_type.csv')
if 'resource_type' not in TANGSHAN:
    import csv
    if isinstance(resource_type, np.ndarray) or isinstance(resource_type,
        pd.DataFrame) or isinstance(resource_type, pd.Series):
        shape_size = resource_type.shape
    elif isinstance(resource_type, list):
        shape_size = len(resource_type)
    else:
        shape_size = 'any'
    check_type = type(resource_type)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('resource_type')
        writer = csv.writer(f)
        writer.writerow(['resource_type', 27, check_type, shape_size])
severity_type = pd.read_csv('../input/severity_type.csv')
if 'severity_type' not in TANGSHAN:
    import csv
    if isinstance(severity_type, np.ndarray) or isinstance(severity_type,
        pd.DataFrame) or isinstance(severity_type, pd.Series):
        shape_size = severity_type.shape
    elif isinstance(severity_type, list):
        shape_size = len(severity_type)
    else:
        shape_size = 'any'
    check_type = type(severity_type)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('severity_type')
        writer = csv.writer(f)
        writer.writerow(['severity_type', 28, check_type, shape_size])
sample_submission = pd.read_csv('../input/sample_submission.csv')
if 'sample_submission' not in TANGSHAN:
    import csv
    if isinstance(sample_submission, np.ndarray) or isinstance(
        sample_submission, pd.DataFrame) or isinstance(sample_submission,
        pd.Series):
        shape_size = sample_submission.shape
    elif isinstance(sample_submission, list):
        shape_size = len(sample_submission)
    else:
        shape_size = 'any'
    check_type = type(sample_submission)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('sample_submission')
        writer = csv.writer(f)
        writer.writerow(['sample_submission', 29, check_type, shape_size])
train.head()
num_train_data = train.shape[0]
enc = preprocessing.OneHotEncoder(sparse=False)
Y = enc.fit_transform(train[['fault_severity']].as_matrix())
if 'enc' not in TANGSHAN:
    import csv
    if isinstance(enc, np.ndarray) or isinstance(enc, pd.DataFrame
        ) or isinstance(enc, pd.Series):
        shape_size = enc.shape
    elif isinstance(enc, list):
        shape_size = len(enc)
    else:
        shape_size = 'any'
    check_type = type(enc)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('enc')
        writer = csv.writer(f)
        writer.writerow(['enc', 42, check_type, shape_size])
train_test = train.drop(['fault_severity'], axis=1)
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
        writer.writerow(['train', 44, check_type, shape_size])
train_test = train_test.append(test, ignore_index=True)
train_test['location'] = train_test.location.map(lambda x: re.findall(
    '\\d+', x)[0])
if 'train_test' not in TANGSHAN:
    import csv
    if isinstance(train_test, np.ndarray) or isinstance(train_test, pd.
        DataFrame) or isinstance(train_test, pd.Series):
        shape_size = train_test.shape
    elif isinstance(train_test, list):
        shape_size = len(train_test)
    else:
        shape_size = 'any'
    check_type = type(train_test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_test')
        writer = csv.writer(f)
        writer.writerow(['train_test', 46, check_type, shape_size])
data = event_type.merge(resource_type, on='id', how='inner')
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
        writer.writerow(['data', 48, check_type, shape_size])
data = data.merge(severity_type, on='id', how='inner')
data = data.merge(log_feature, on='id', how='inner')
data['severity_type'] = data.severity_type.map(lambda x: re.findall('\\d+',
    x)[0])
data['log_feature'] = data.log_feature.map(lambda x: re.findall('\\d+', x)[0])
data['event_type'] = data.event_type.map(lambda x: re.findall('\\d+', x)[0])
data['resource_type'] = data.resource_type.map(lambda x: re.findall('\\d+',
    x)[0])
if 'x' not in TANGSHAN:
    import csv
    if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame) or isinstance(x
        , pd.Series):
        shape_size = x.shape
    elif isinstance(x, list):
        shape_size = len(x)
    else:
        shape_size = 'any'
    check_type = type(x)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x')
        writer = csv.writer(f)
        writer.writerow(['x', 54, check_type, shape_size])
all_data = pd.merge(train_test, data, on='id', how='inner')
all_data.head(10)
feature_name = ['location', 'event_type', 'resource_type', 'severity_type',
    'log_feature']
if 'feature_name' not in TANGSHAN:
    import csv
    if isinstance(feature_name, np.ndarray) or isinstance(feature_name, pd.
        DataFrame) or isinstance(feature_name, pd.Series):
        shape_size = feature_name.shape
    elif isinstance(feature_name, list):
        shape_size = len(feature_name)
    else:
        shape_size = 'any'
    check_type = type(feature_name)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('feature_name')
        writer = csv.writer(f)
        writer.writerow(['feature_name', 68, check_type, shape_size])
enc = preprocessing.OneHotEncoder(n_values=np.array([1127, 55, 11, 6, 387]),
    sparse=False)
features = all_data['id']
if 'all_data' not in TANGSHAN:
    import csv
    if isinstance(all_data, np.ndarray) or isinstance(all_data, pd.DataFrame
        ) or isinstance(all_data, pd.Series):
        shape_size = all_data.shape
    elif isinstance(all_data, list):
        shape_size = len(all_data)
    else:
        shape_size = 'any'
    check_type = type(all_data)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('all_data')
        writer = csv.writer(f)
        writer.writerow(['all_data', 70, check_type, shape_size])
features = pd.concat([features, pd.DataFrame(data=enc.fit_transform(
    all_data[feature_name].as_matrix()))], axis=1)
features = features.astype(int)
if 'int' not in TANGSHAN:
    import csv
    if isinstance(int, np.ndarray) or isinstance(int, pd.DataFrame
        ) or isinstance(int, pd.Series):
        shape_size = int.shape
    elif isinstance(int, list):
        shape_size = len(int)
    else:
        shape_size = 'any'
    check_type = type(int)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('int')
        writer = csv.writer(f)
        writer.writerow(['int', 72, check_type, shape_size])
features['vol'] = all_data['volume']
feature_name.append('volume')
features = features.groupby(['id'], sort=False, as_index=False).sum()
features.head()
features_norm = (features - features.mean()) / (features.max() - features.min()
    )
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
        writer.writerow(['features', 86, check_type, shape_size])
if 'features_norm' not in TANGSHAN:
    import csv
    if isinstance(features_norm, np.ndarray) or isinstance(features_norm,
        pd.DataFrame) or isinstance(features_norm, pd.Series):
        shape_size = features_norm.shape
    elif isinstance(features_norm, list):
        shape_size = len(features_norm)
    else:
        shape_size = 'any'
    check_type = type(features_norm)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('features_norm')
        writer = csv.writer(f)
        writer.writerow(['features_norm', 86, check_type, shape_size])
cross_corr = features_norm[:num_train_data].drop(['id'], axis=1)
cross_corr['Y0'] = Y[:, (0)]
cross_corr['Y1'] = Y[:, (1)]
cross_corr['Y2'] = Y[:, (2)]
if 'Y' not in TANGSHAN:
    import csv
    if isinstance(Y, np.ndarray) or isinstance(Y, pd.DataFrame) or isinstance(Y
        , pd.Series):
        shape_size = Y.shape
    elif isinstance(Y, list):
        shape_size = len(Y)
    else:
        shape_size = 'any'
    check_type = type(Y)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('Y')
        writer = csv.writer(f)
        writer.writerow(['Y', 90, check_type, shape_size])
cross_corr = cross_corr.corr().fillna(0)
cross_corr = cross_corr[['Y0', 'Y1', 'Y2']].drop(['Y0', 'Y1', 'Y2'])
cross_corr.transpose()
clr = ['pink', 'red', 'blue', 'yellow', 'green', 'black']
if 'clr' not in TANGSHAN:
    import csv
    if isinstance(clr, np.ndarray) or isinstance(clr, pd.DataFrame
        ) or isinstance(clr, pd.Series):
        shape_size = clr.shape
    elif isinstance(clr, list):
        shape_size = len(clr)
    else:
        shape_size = 'any'
    check_type = type(clr)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('clr')
        writer = csv.writer(f)
        writer.writerow(['clr', 98, check_type, shape_size])
for idx_corr in range(3):
    Y_corr = cross_corr.iloc[:, (idx_corr)]
    if 'idx_corr' not in TANGSHAN:
        import csv
        if isinstance(idx_corr, np.ndarray) or isinstance(idx_corr, pd.
            DataFrame) or isinstance(idx_corr, pd.Series):
            shape_size = idx_corr.shape
        elif isinstance(idx_corr, list):
            shape_size = len(idx_corr)
        else:
            shape_size = 'any'
        check_type = type(idx_corr)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('idx_corr')
            writer = csv.writer(f)
            writer.writerow(['idx_corr', 101, check_type, shape_size])
    if 'cross_corr' not in TANGSHAN:
        import csv
        if isinstance(cross_corr, np.ndarray) or isinstance(cross_corr, pd.
            DataFrame) or isinstance(cross_corr, pd.Series):
            shape_size = cross_corr.shape
        elif isinstance(cross_corr, list):
            shape_size = len(cross_corr)
        else:
            shape_size = 'any'
        check_type = type(cross_corr)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('cross_corr')
            writer = csv.writer(f)
            writer.writerow(['cross_corr', 101, check_type, shape_size])
    plt.figure(figsize=(15, 6))
    for idx, n_featuer in enumerate(enc.n_values):
        xx = np.arange(enc.feature_indices_[idx], enc.feature_indices_[idx + 1]
            )
        if 'idx' not in TANGSHAN:
            import csv
            if isinstance(idx, np.ndarray) or isinstance(idx, pd.DataFrame
                ) or isinstance(idx, pd.Series):
                shape_size = idx.shape
            elif isinstance(idx, list):
                shape_size = len(idx)
            else:
                shape_size = 'any'
            check_type = type(idx)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('idx')
                writer = csv.writer(f)
                writer.writerow(['idx', 104, check_type, shape_size])
        if 'xx' not in TANGSHAN:
            import csv
            if isinstance(xx, np.ndarray) or isinstance(xx, pd.DataFrame
                ) or isinstance(xx, pd.Series):
                shape_size = xx.shape
            elif isinstance(xx, list):
                shape_size = len(xx)
            else:
                shape_size = 'any'
            check_type = type(xx)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xx')
                writer = csv.writer(f)
                writer.writerow(['xx', 104, check_type, shape_size])
        yy = Y_corr[xx]
        if 'Y_corr' not in TANGSHAN:
            import csv
            if isinstance(Y_corr, np.ndarray) or isinstance(Y_corr, pd.
                DataFrame) or isinstance(Y_corr, pd.Series):
                shape_size = Y_corr.shape
            elif isinstance(Y_corr, list):
                shape_size = len(Y_corr)
            else:
                shape_size = 'any'
            check_type = type(Y_corr)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('Y_corr')
                writer = csv.writer(f)
                writer.writerow(['Y_corr', 105, check_type, shape_size])
        plt.bar(xx, yy, width=5, color=clr[idx])
        if 'yy' not in TANGSHAN:
            import csv
            if isinstance(yy, np.ndarray) or isinstance(yy, pd.DataFrame
                ) or isinstance(yy, pd.Series):
                shape_size = yy.shape
            elif isinstance(yy, list):
                shape_size = len(yy)
            else:
                shape_size = 'any'
            check_type = type(yy)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('yy')
                writer = csv.writer(f)
                writer.writerow(['yy', 106, check_type, shape_size])
        plt.bar(1586, Y_corr[1586], width, color=clr[idx + 1])
        if 'width' not in TANGSHAN:
            import csv
            if isinstance(width, np.ndarray) or isinstance(width, pd.DataFrame
                ) or isinstance(width, pd.Series):
                shape_size = width.shape
            elif isinstance(width, list):
                shape_size = len(width)
            else:
                shape_size = 'any'
            check_type = type(width)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('width')
                writer = csv.writer(f)
                writer.writerow(['width', 107, check_type, shape_size])
        plt.xlabel('Feature Number')
        plt.ylabel('Feature Cross-Correlation to Fault Severity ' + str(
            idx_corr))
        plt.yticks(np.arange(min(yy), max(yy), 0.02))
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
                writer.writerow(['min', 110, check_type, shape_size])
        if 'max' not in TANGSHAN:
            import csv
            if isinstance(max, np.ndarray) or isinstance(max, pd.DataFrame
                ) or isinstance(max, pd.Series):
                shape_size = max.shape
            elif isinstance(max, list):
                shape_size = len(max)
            else:
                shape_size = 'any'
            check_type = type(max)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('max')
                writer = csv.writer(f)
                writer.writerow(['max', 110, check_type, shape_size])
        plt.grid(which='both', axis='both')
        plt.legend(feature_name, bbox_to_anchor=(0.17, 1))
    if 'enumerate' not in TANGSHAN:
        import csv
        if isinstance(enumerate, np.ndarray) or isinstance(enumerate, pd.
            DataFrame) or isinstance(enumerate, pd.Series):
            shape_size = enumerate.shape
        elif isinstance(enumerate, list):
            shape_size = len(enumerate)
        else:
            shape_size = 'any'
        check_type = type(enumerate)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('enumerate')
            writer = csv.writer(f)
            writer.writerow(['enumerate', 103, check_type, shape_size])
    if 'n_featuer' not in TANGSHAN:
        import csv
        if isinstance(n_featuer, np.ndarray) or isinstance(n_featuer, pd.
            DataFrame) or isinstance(n_featuer, pd.Series):
            shape_size = n_featuer.shape
        elif isinstance(n_featuer, list):
            shape_size = len(n_featuer)
        else:
            shape_size = 'any'
        check_type = type(n_featuer)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('n_featuer')
            writer = csv.writer(f)
            writer.writerow(['n_featuer', 103, check_type, shape_size])
cct = 0.03
if 'cct' not in TANGSHAN:
    import csv
    if isinstance(cct, np.ndarray) or isinstance(cct, pd.DataFrame
        ) or isinstance(cct, pd.Series):
        shape_size = cct.shape
    elif isinstance(cct, list):
        shape_size = len(cct)
    else:
        shape_size = 'any'
    check_type = type(cct)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cct')
        writer = csv.writer(f)
        writer.writerow(['cct', 124, check_type, shape_size])
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
for idx in range(3):
    X = features_norm[:num_train_data].drop(['id'], axis=1).as_matrix()
    X = X[:, (np.transpose(np.nonzero(np.absolute(cross_corr.iloc[:, (idx)]
        ) > cct)))][:, :, (0)]
    if 'X' not in TANGSHAN:
        import csv
        if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame
            ) or isinstance(X, pd.Series):
            shape_size = X.shape
        elif isinstance(X, list):
            shape_size = len(X)
        else:
            shape_size = 'any'
        check_type = type(X)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('X')
            writer = csv.writer(f)
            writer.writerow(['X', 135, check_type, shape_size])
    gb = GradientBoostingClassifier()
    if 'gb' not in TANGSHAN:
        import csv
        if isinstance(gb, np.ndarray) or isinstance(gb, pd.DataFrame
            ) or isinstance(gb, pd.Series):
            shape_size = gb.shape
        elif isinstance(gb, list):
            shape_size = len(gb)
        else:
            shape_size = 'any'
        check_type = type(gb)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('gb')
            writer = csv.writer(f)
            writer.writerow(['gb', 136, check_type, shape_size])
    clf = GridSearchCV(gb, {'max_depth': [2, 3, 4], 'n_estimators': [100, 
        150], 'learning_rate': [0.5], 'subsample': [1], 'max_leaf_nodes': [
        3]}, verbose=1, n_jobs=2, cv=3, scoring='log_loss')
    clf.fit(X, Y[:, (idx)])
    joblib.dump(clf, 'D:/My Completed Downloads/Telestra/clfgb' + str(idx) +
        '.pkl')
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
        writer.writerow(['range', 133, check_type, shape_size])
Y_pred = []
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
        writer.writerow(['Y_pred', 146, check_type, shape_size])
log_loss = []
if 'log_loss' not in TANGSHAN:
    import csv
    if isinstance(log_loss, np.ndarray) or isinstance(log_loss, pd.DataFrame
        ) or isinstance(log_loss, pd.Series):
        shape_size = log_loss.shape
    elif isinstance(log_loss, list):
        shape_size = len(log_loss)
    else:
        shape_size = 'any'
    check_type = type(log_loss)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('log_loss')
        writer = csv.writer(f)
        writer.writerow(['log_loss', 147, check_type, shape_size])
for idx in range(3):
    clf = joblib.load('D:/My Completed Downloads/Telestra/clfgb' + str(idx) +
        '.pkl')
    X_t = features_norm[num_train_data:].drop(['id'], axis=1).as_matrix()
    if 'num_train_data' not in TANGSHAN:
        import csv
        if isinstance(num_train_data, np.ndarray) or isinstance(num_train_data,
            pd.DataFrame) or isinstance(num_train_data, pd.Series):
            shape_size = num_train_data.shape
        elif isinstance(num_train_data, list):
            shape_size = len(num_train_data)
        else:
            shape_size = 'any'
        check_type = type(num_train_data)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('num_train_data')
            writer = csv.writer(f)
            writer.writerow(['num_train_data', 150, check_type, shape_size])
    X_t = X_t[:, (np.transpose(np.nonzero(np.absolute(cross_corr.iloc[:, (
        idx)]) > cct)))][:, :, (0)]
    Y_pred.append(clf.predict_proba(X_t))
    log_loss.append(-clf.best_score_)
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
            writer.writerow(['clf', 153, check_type, shape_size])
    print('Number of features for model ' + str(idx) + ' : ' + str(X_t.
        shape[1]))
    if 'X_t' not in TANGSHAN:
        import csv
        if isinstance(X_t, np.ndarray) or isinstance(X_t, pd.DataFrame
            ) or isinstance(X_t, pd.Series):
            shape_size = X_t.shape
        elif isinstance(X_t, list):
            shape_size = len(X_t)
        else:
            shape_size = 'any'
        check_type = type(X_t)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('X_t')
            writer = csv.writer(f)
            writer.writerow(['X_t', 154, check_type, shape_size])
print('Model log-loss: ' + str(np.mean(log_loss)))
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
        writer.writerow(['print', 156, check_type, shape_size])
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
        writer.writerow(['str', 156, check_type, shape_size])
prediction = pd.DataFrame({'id': features[num_train_data:].id, 'predict_0':
    Y_pred[0][:, (1)], 'predict_1': Y_pred[1][:, (1)], 'predict_2': Y_pred[
    2][:, (1)]})
prediction.to_csv('D:/My Completed Downloads/Telestra/Telestra_gbclf.csv',
    index=False)
if 'prediction' not in TANGSHAN:
    import csv
    if isinstance(prediction, np.ndarray) or isinstance(prediction, pd.
        DataFrame) or isinstance(prediction, pd.Series):
        shape_size = prediction.shape
    elif isinstance(prediction, list):
        shape_size = len(prediction)
    else:
        shape_size = 'any'
    check_type = type(prediction)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('prediction')
        writer = csv.writer(f)
        writer.writerow(['prediction', 161, check_type, shape_size])
prediction.head()
