import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn
import xgboost as xgb
from math import log
dataframe = pd.read_csv('../input/train.csv')
LE = LabelEncoder()
DV = DictVectorizer()
if 'DV' not in TANGSHAN:
    import csv
    if isinstance(DV, np.ndarray) or isinstance(DV, pd.DataFrame
        ) or isinstance(DV, pd.Series):
        shape_size = DV.shape
    elif isinstance(DV, list):
        shape_size = len(DV)
    else:
        shape_size = 'any'
    check_type = type(DV)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('DV')
        writer = csv.writer(f)
        writer.writerow(['DV', 34, check_type, shape_size])
data = np.log(2.0 + DV.fit_transform(dataframe.iloc[:, 1:94].T.to_dict().
    values()).todense())
LE.fit(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
    'Class_7', 'Class_8', 'Class_9'])
labels = LE.transform(dataframe.iloc[:, (94)])
if 'LE' not in TANGSHAN:
    import csv
    if isinstance(LE, np.ndarray) or isinstance(LE, pd.DataFrame
        ) or isinstance(LE, pd.Series):
        shape_size = LE.shape
    elif isinstance(LE, list):
        shape_size = len(LE)
    else:
        shape_size = 'any'
    check_type = type(LE)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('LE')
        writer = csv.writer(f)
        writer.writerow(['LE', 39, check_type, shape_size])
idx = np.array(dataframe.iloc[:, (0)])
if 'dataframe' not in TANGSHAN:
    import csv
    if isinstance(dataframe, np.ndarray) or isinstance(dataframe, pd.DataFrame
        ) or isinstance(dataframe, pd.Series):
        shape_size = dataframe.shape
    elif isinstance(dataframe, list):
        shape_size = len(dataframe)
    else:
        shape_size = 'any'
    check_type = type(dataframe)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dataframe')
        writer = csv.writer(f)
        writer.writerow(['dataframe', 40, check_type, shape_size])
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
        writer.writerow(['idx', 40, check_type, shape_size])
shuf = np.random.permutation(labels.shape[0])
idx = idx[shuf]
if 'shuf' not in TANGSHAN:
    import csv
    if isinstance(shuf, np.ndarray) or isinstance(shuf, pd.DataFrame
        ) or isinstance(shuf, pd.Series):
        shape_size = shuf.shape
    elif isinstance(shuf, list):
        shape_size = len(shuf)
    else:
        shape_size = 'any'
    check_type = type(shuf)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('shuf')
        writer = csv.writer(f)
        writer.writerow(['shuf', 44, check_type, shape_size])
labels = labels[shuf]
data = data[(shuf), :]
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
        writer.writerow(['data', 46, check_type, shape_size])
param = {'max_depth': 10, 'min_loss_reduction': 0.6, 'min_child_weight': 6,
    'subsample': 0.7, 'eta': 0.3, 'silent': 1, 'objective':
    'multi:softprob', 'nthread': 4, 'num_class': 9}
if 'param' not in TANGSHAN:
    import csv
    if isinstance(param, np.ndarray) or isinstance(param, pd.DataFrame
        ) or isinstance(param, pd.Series):
        shape_size = param.shape
    elif isinstance(param, list):
        shape_size = len(param)
    else:
        shape_size = 'any'
    check_type = type(param)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('param')
        writer = csv.writer(f)
        writer.writerow(['param', 50, check_type, shape_size])
SKF = StratifiedKFold(n_folds=3, y=labels)
if 'SKF' not in TANGSHAN:
    import csv
    if isinstance(SKF, np.ndarray) or isinstance(SKF, pd.DataFrame
        ) or isinstance(SKF, pd.Series):
        shape_size = SKF.shape
    elif isinstance(SKF, list):
        shape_size = len(SKF)
    else:
        shape_size = 'any'
    check_type = type(SKF)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('SKF')
        writer = csv.writer(f)
        writer.writerow(['SKF', 52, check_type, shape_size])
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
        writer.writerow(['labels', 52, check_type, shape_size])
train_preds = np.zeros((0, 9))
for tridx, tsidx in SKF:
    dtrain = xgb.DMatrix(data[tridx], label=labels[tridx])
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
            writer.writerow(['dtrain', 56, check_type, shape_size])
    dtest = xgb.DMatrix(data[tsidx])
    if 'tsidx' not in TANGSHAN:
        import csv
        if isinstance(tsidx, np.ndarray) or isinstance(tsidx, pd.DataFrame
            ) or isinstance(tsidx, pd.Series):
            shape_size = tsidx.shape
        elif isinstance(tsidx, list):
            shape_size = len(tsidx)
        else:
            shape_size = 'any'
        check_type = type(tsidx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('tsidx')
            writer = csv.writer(f)
            writer.writerow(['tsidx', 57, check_type, shape_size])
    bst = xgb.train(param, dtrain, 20)
    if 'bst' not in TANGSHAN:
        import csv
        if isinstance(bst, np.ndarray) or isinstance(bst, pd.DataFrame
            ) or isinstance(bst, pd.Series):
            shape_size = bst.shape
        elif isinstance(bst, list):
            shape_size = len(bst)
        else:
            shape_size = 'any'
        check_type = type(bst)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('bst')
            writer = csv.writer(f)
            writer.writerow(['bst', 58, check_type, shape_size])
    train_preds = np.append(train_preds, bst.predict(dtest), axis=0)
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
            writer.writerow(['dtest', 59, check_type, shape_size])
    if 'train_preds' not in TANGSHAN:
        import csv
        if isinstance(train_preds, np.ndarray) or isinstance(train_preds,
            pd.DataFrame) or isinstance(train_preds, pd.Series):
            shape_size = train_preds.shape
        elif isinstance(train_preds, list):
            shape_size = len(train_preds)
        else:
            shape_size = 'any'
        check_type = type(train_preds)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train_preds')
            writer = csv.writer(f)
            writer.writerow(['train_preds', 59, check_type, shape_size])
if 'tridx' not in TANGSHAN:
    import csv
    if isinstance(tridx, np.ndarray) or isinstance(tridx, pd.DataFrame
        ) or isinstance(tridx, pd.Series):
        shape_size = tridx.shape
    elif isinstance(tridx, list):
        shape_size = len(tridx)
    else:
        shape_size = 'any'
    check_type = type(tridx)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tridx')
        writer = csv.writer(f)
        writer.writerow(['tridx', 55, check_type, shape_size])
err = 0
for i in range(train_preds.shape[0]):
    err -= log(1e-15 + train_preds[i, labels[i]])
if 'i' not in TANGSHAN:
    import csv
    if isinstance(i, np.ndarray) or isinstance(i, pd.DataFrame) or isinstance(i
        , pd.Series):
        shape_size = i.shape
    elif isinstance(i, list):
        shape_size = len(i)
    else:
        shape_size = 'any'
    check_type = type(i)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('i')
        writer = csv.writer(f)
        writer.writerow(['i', 62, check_type, shape_size])
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
        writer.writerow(['range', 62, check_type, shape_size])
err /= train_preds.shape[0]
if 'err' not in TANGSHAN:
    import csv
    if isinstance(err, np.ndarray) or isinstance(err, pd.DataFrame
        ) or isinstance(err, pd.Series):
        shape_size = err.shape
    elif isinstance(err, list):
        shape_size = len(err)
    else:
        shape_size = 'any'
    check_type = type(err)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('err')
        writer = csv.writer(f)
        writer.writerow(['err', 65, check_type, shape_size])
print('CV log-loss of the base estimator: {}'.format(err))
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
        writer.writerow(['print', 67, check_type, shape_size])
for examine_class in range(9):
    threshold = 0.3
    baseidx = labels != examine_class
    wrongidx = train_preds[baseidx, examine_class] >= threshold
    if 'wrongidx' not in TANGSHAN:
        import csv
        if isinstance(wrongidx, np.ndarray) or isinstance(wrongidx, pd.
            DataFrame) or isinstance(wrongidx, pd.Series):
            shape_size = wrongidx.shape
        elif isinstance(wrongidx, list):
            shape_size = len(wrongidx)
        else:
            shape_size = 'any'
        check_type = type(wrongidx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('wrongidx')
            writer = csv.writer(f)
            writer.writerow(['wrongidx', 78, check_type, shape_size])
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
            writer.writerow(['threshold', 78, check_type, shape_size])
    rightidx = train_preds[baseidx, examine_class] < threshold
    if 'baseidx' not in TANGSHAN:
        import csv
        if isinstance(baseidx, np.ndarray) or isinstance(baseidx, pd.DataFrame
            ) or isinstance(baseidx, pd.Series):
            shape_size = baseidx.shape
        elif isinstance(baseidx, list):
            shape_size = len(baseidx)
        else:
            shape_size = 'any'
        check_type = type(baseidx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('baseidx')
            writer = csv.writer(f)
            writer.writerow(['baseidx', 80, check_type, shape_size])
    wrongdata = data[(baseidx), :][(wrongidx), :]
    if 'wrongdata' not in TANGSHAN:
        import csv
        if isinstance(wrongdata, np.ndarray) or isinstance(wrongdata, pd.
            DataFrame) or isinstance(wrongdata, pd.Series):
            shape_size = wrongdata.shape
        elif isinstance(wrongdata, list):
            shape_size = len(wrongdata)
        else:
            shape_size = 'any'
        check_type = type(wrongdata)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('wrongdata')
            writer = csv.writer(f)
            writer.writerow(['wrongdata', 82, check_type, shape_size])
    okaydata = data[(baseidx), :][(rightidx), :]
    if 'rightidx' not in TANGSHAN:
        import csv
        if isinstance(rightidx, np.ndarray) or isinstance(rightidx, pd.
            DataFrame) or isinstance(rightidx, pd.Series):
            shape_size = rightidx.shape
        elif isinstance(rightidx, list):
            shape_size = len(rightidx)
        else:
            shape_size = 'any'
        check_type = type(rightidx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('rightidx')
            writer = csv.writer(f)
            writer.writerow(['rightidx', 83, check_type, shape_size])
    lda = LDA()
    pca = PCA()
    if 'pca' not in TANGSHAN:
        import csv
        if isinstance(pca, np.ndarray) or isinstance(pca, pd.DataFrame
            ) or isinstance(pca, pd.Series):
            shape_size = pca.shape
        elif isinstance(pca, list):
            shape_size = len(pca)
        else:
            shape_size = 'any'
        check_type = type(pca)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pca')
            writer = csv.writer(f)
            writer.writerow(['pca', 88, check_type, shape_size])
    wrongproj = pca.fit_transform(wrongdata)
    okayproj = pca.transform(okaydata)
    wl = np.zeros(wrongproj.shape[0])
    if 'wl' not in TANGSHAN:
        import csv
        if isinstance(wl, np.ndarray) or isinstance(wl, pd.DataFrame
            ) or isinstance(wl, pd.Series):
            shape_size = wl.shape
        elif isinstance(wl, list):
            shape_size = len(wl)
        else:
            shape_size = 'any'
        check_type = type(wl)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('wl')
            writer = csv.writer(f)
            writer.writerow(['wl', 94, check_type, shape_size])
    wl[:] = 1
    ol = np.zeros(okayproj.shape[0])
    ol[:] = 0
    if 'ol' not in TANGSHAN:
        import csv
        if isinstance(ol, np.ndarray) or isinstance(ol, pd.DataFrame
            ) or isinstance(ol, pd.Series):
            shape_size = ol.shape
        elif isinstance(ol, list):
            shape_size = len(ol)
        else:
            shape_size = 'any'
        check_type = type(ol)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('ol')
            writer = csv.writer(f)
            writer.writerow(['ol', 97, check_type, shape_size])
    pcl = np.append(wl, ol)
    if 'pcl' not in TANGSHAN:
        import csv
        if isinstance(pcl, np.ndarray) or isinstance(pcl, pd.DataFrame
            ) or isinstance(pcl, pd.Series):
            shape_size = pcl.shape
        elif isinstance(pcl, list):
            shape_size = len(pcl)
        else:
            shape_size = 'any'
        check_type = type(pcl)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pcl')
            writer = csv.writer(f)
            writer.writerow(['pcl', 98, check_type, shape_size])
    lda.fit(np.append(wrongdata, okaydata, axis=0), pcl)
    if 'okaydata' not in TANGSHAN:
        import csv
        if isinstance(okaydata, np.ndarray) or isinstance(okaydata, pd.
            DataFrame) or isinstance(okaydata, pd.Series):
            shape_size = okaydata.shape
        elif isinstance(okaydata, list):
            shape_size = len(okaydata)
        else:
            shape_size = 'any'
        check_type = type(okaydata)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('okaydata')
            writer = csv.writer(f)
            writer.writerow(['okaydata', 100, check_type, shape_size])
    wrong_ld = lda.transform(wrongdata)
    okay_ld = lda.transform(okaydata)
    if 'okay_ld' not in TANGSHAN:
        import csv
        if isinstance(okay_ld, np.ndarray) or isinstance(okay_ld, pd.DataFrame
            ) or isinstance(okay_ld, pd.Series):
            shape_size = okay_ld.shape
        elif isinstance(okay_ld, list):
            shape_size = len(okay_ld)
        else:
            shape_size = 'any'
        check_type = type(okay_ld)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('okay_ld')
            writer = csv.writer(f)
            writer.writerow(['okay_ld', 102, check_type, shape_size])
    if 'lda' not in TANGSHAN:
        import csv
        if isinstance(lda, np.ndarray) or isinstance(lda, pd.DataFrame
            ) or isinstance(lda, pd.Series):
            shape_size = lda.shape
        elif isinstance(lda, list):
            shape_size = len(lda)
        else:
            shape_size = 'any'
        check_type = type(lda)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('lda')
            writer = csv.writer(f)
            writer.writerow(['lda', 102, check_type, shape_size])
    plt.plot(okay_ld[:, (0)], okayproj[:, (0)], 'o', markersize=4, alpha=
        0.2, label='Not Mistaken', color='#3030a0')
    plt.plot(wrong_ld[:, (0)], wrongproj[:, (0)], 'o', markersize=6, label=
        'Mistaken', alpha=0.8, color='#f03030', markeredgecolor='#600000',
        markeredgewidth=1)
    if 'wrong_ld' not in TANGSHAN:
        import csv
        if isinstance(wrong_ld, np.ndarray) or isinstance(wrong_ld, pd.
            DataFrame) or isinstance(wrong_ld, pd.Series):
            shape_size = wrong_ld.shape
        elif isinstance(wrong_ld, list):
            shape_size = len(wrong_ld)
        else:
            shape_size = 'any'
        check_type = type(wrong_ld)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('wrong_ld')
            writer = csv.writer(f)
            writer.writerow(['wrong_ld', 105, check_type, shape_size])
    plt.xlabel('LDA Feature')
    plt.ylabel('PCA Feature')
    plt.ylim(min(okayproj[:, (0)].min(), wrongproj[:, (0)].min()) - 1, max(
        okayproj[:, (0)].max(), wrongproj[:, (0)].max()) + 2)
    if 'wrongproj' not in TANGSHAN:
        import csv
        if isinstance(wrongproj, np.ndarray) or isinstance(wrongproj, pd.
            DataFrame) or isinstance(wrongproj, pd.Series):
            shape_size = wrongproj.shape
        elif isinstance(wrongproj, list):
            shape_size = len(wrongproj)
        else:
            shape_size = 'any'
        check_type = type(wrongproj)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('wrongproj')
            writer = csv.writer(f)
            writer.writerow(['wrongproj', 108, check_type, shape_size])
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
            writer.writerow(['min', 108, check_type, shape_size])
    if 'okayproj' not in TANGSHAN:
        import csv
        if isinstance(okayproj, np.ndarray) or isinstance(okayproj, pd.
            DataFrame) or isinstance(okayproj, pd.Series):
            shape_size = okayproj.shape
        elif isinstance(okayproj, list):
            shape_size = len(okayproj)
        else:
            shape_size = 'any'
        check_type = type(okayproj)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('okayproj')
            writer = csv.writer(f)
            writer.writerow(['okayproj', 108, check_type, shape_size])
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
            writer.writerow(['max', 108, check_type, shape_size])
    plt.title('Misclassification as Class ' + str(examine_class + 1))
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
            writer.writerow(['str', 109, check_type, shape_size])
    plt.legend()
    plt.savefig('class' + str(examine_class + 1) + '.png')
    if 'examine_class' not in TANGSHAN:
        import csv
        if isinstance(examine_class, np.ndarray) or isinstance(examine_class,
            pd.DataFrame) or isinstance(examine_class, pd.Series):
            shape_size = examine_class.shape
        elif isinstance(examine_class, list):
            shape_size = len(examine_class)
        else:
            shape_size = 'any'
        check_type = type(examine_class)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('examine_class')
            writer = csv.writer(f)
            writer.writerow(['examine_class', 111, check_type, shape_size])
    plt.clf()
