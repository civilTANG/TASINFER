import pandas as pd
import numpy as np
import xgboost as xgb
import zipfile


def get_num_radar_scans(timetoend_row):
    return timetoend_row.count(' ') + 1


def mean_of_row(row):
    return np.mean(list(map(np.double, row.split(' '))))


def make_cdf_step(true_label_value):
    step_cdf = np.ones(70)
    step_cdf[0:true_label_value] = 0
    return step_cdf


def make_cdf_distribution(in_class_labels):
    pdf = in_class_labels.value_counts() / float(len(in_class_labels))
    pdf = pdf.sort_index()
    cdf = np.zeros(70)
    for e, i in enumerate(pdf.index.values.tolist()):
        cdf[i] = pdf.iloc[e]
    return cdf.cumsum()


def make_cdf_list(first_agg, num_lab, new_lab, actual_labels, offset):
    cdfs = []
    for i in range(num_lab):
        if i < first_agg:
            cdfs.append(make_cdf_step(i))
        else:
            cdfs.append(make_cdf_distribution(actual_labels.reindex(new_lab
                .iloc[offset:][new_lab.iloc[offset:] == i].index)))
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
            writer.writerow(['range', 33, check_type, shape_size])
    return cdfs


def create_full_predictions(CDFs, predictions):
    data_length = len(predictions)
    for e, i in enumerate(CDFs):
        if e == 0:
            temp = predictions.iloc[:, (0)].values.reshape(data_length, 1
                ) * CDFs[0].reshape(1, len(CDFs[0]))
        else:
            temp += predictions.iloc[:, (e)].values.reshape(data_length, 1
                ) * CDFs[e].reshape(1, len(CDFs[e]))
    return temp


def aggregate_labels(label_list, ceiled_labels):
    new_lab = ceiled_labels.replace(label_list[0][0], label_list[0][1])
    for i in range(1, len(label_list)):
        new_lab = new_lab.replace(label_list[i][0], label_list[i][1])
    return new_lab


def train_linear_xgb(data, lmbda, alpha, lmbda_bias, num_classes,
    num_threads, num_rounds, early_stop=3):
    xg_train = xgb.DMatrix(data[0].values, label=data[1].values.ravel(),
        missing=np.nan)
    xg_val = xgb.DMatrix(data[2].values, label=data[3].values.ravel(),
        missing=np.nan)
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
            writer.writerow(['data', 59, check_type, shape_size])
    param1 = {}
    param1['objective'] = 'multi:softprob'
    param1['lambda'] = lmbda
    param1['alpha'] = alpha
    param1['lambda_bias'] = lmbda_bias
    param1['silent'] = 1
    param1['nthread'] = num_threads
    param1['num_class'] = num_classes
    param1['eval_metric'] = 'mlogloss'
    watchlist = [(xg_train, 'train'), (xg_val, 'test')]
    bst1 = xgb.train(param1, xg_train, num_rounds, evals=watchlist,
        early_stopping_rounds=early_stop)
    return bst1


def predict_bst(bst, validation):
    xg_val = xgb.DMatrix(validation.values, missing=np.nan)
    pred = bst.predict(xg_val)
    pred = pd.DataFrame(pred, index=validation.index)
    return pred


def calc_crps(thresholds, predictions, actuals):
    obscdf = (thresholds.reshape(70, 1) >= actuals).T
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps


z = zipfile.ZipFile('../input/train_2013.csv.zip')
if 'z' not in TANGSHAN:
    import csv
    if isinstance(z, np.ndarray) or isinstance(z, pd.DataFrame) or isinstance(z
        , pd.Series):
        shape_size = z.shape
    elif isinstance(z, list):
        shape_size = len(z)
    else:
        shape_size = 'any'
    check_type = type(z)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('z')
        writer = csv.writer(f)
        writer.writerow(['z', 92, check_type, shape_size])
train = pd.read_csv(z.open('train_2013.csv'), usecols=['Expected',
    'Reflectivity'])
train['num_scans'] = train.Reflectivity.apply(get_num_radar_scans)
train = train.query('num_scans > 17')
train['mean_reflectivity'] = train.Reflectivity.apply(mean_of_row)
labels = train.Expected
train.drop(['Reflectivity', 'Expected'], axis=1, inplace=True)
integer_labels = np.ceil(labels)
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
        writer.writerow(['labels', 103, check_type, shape_size])
integer_labels = integer_labels[integer_labels < 70]
reduced_labels = aggregate_labels([[range(8, 10), 8], [range(10, 14), 9], [
    range(14, 19), 10], [range(19, 70), 11]], pd.DataFrame(integer_labels)
    ).iloc[:, (0)]
train = train.reindex(integer_labels.index)
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
        writer.writerow(['train', 107, check_type, shape_size])
cutoff_value = int(len(train) * 0.5)
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
        writer.writerow(['int', 108, check_type, shape_size])
train = pd.concat([train.mean_reflectivity, train.num_scans], axis=1)
data = train.iloc[cutoff_value:, :], reduced_labels.iloc[cutoff_value:
    ], train.iloc[:cutoff_value, :], reduced_labels.iloc[:cutoff_value]
if 'reduced_labels' not in TANGSHAN:
    import csv
    if isinstance(reduced_labels, np.ndarray) or isinstance(reduced_labels,
        pd.DataFrame) or isinstance(reduced_labels, pd.Series):
        shape_size = reduced_labels.shape
    elif isinstance(reduced_labels, list):
        shape_size = len(reduced_labels)
    else:
        shape_size = 'any'
    check_type = type(reduced_labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('reduced_labels')
        writer = csv.writer(f)
        writer.writerow(['reduced_labels', 113, check_type, shape_size])
bst = train_linear_xgb(data, 5, 5, 2, 12, 1, num_rounds=56, early_stop=2)
preds = predict_bst(bst, train.iloc[:cutoff_value])
if 'preds' not in TANGSHAN:
    import csv
    if isinstance(preds, np.ndarray) or isinstance(preds, pd.DataFrame
        ) or isinstance(preds, pd.Series):
        shape_size = preds.shape
    elif isinstance(preds, list):
        shape_size = len(preds)
    else:
        shape_size = 'any'
    check_type = type(preds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('preds')
        writer = csv.writer(f)
        writer.writerow(['preds', 119, check_type, shape_size])
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
        writer.writerow(['bst', 119, check_type, shape_size])
if 'cutoff_value' not in TANGSHAN:
    import csv
    if isinstance(cutoff_value, np.ndarray) or isinstance(cutoff_value, pd.
        DataFrame) or isinstance(cutoff_value, pd.Series):
        shape_size = cutoff_value.shape
    elif isinstance(cutoff_value, list):
        shape_size = len(cutoff_value)
    else:
        shape_size = 'any'
    check_type = type(cutoff_value)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cutoff_value')
        writer = csv.writer(f)
        writer.writerow(['cutoff_value', 119, check_type, shape_size])
cdfs_tst = make_cdf_list(8, 12, reduced_labels, integer_labels, 0)
if 'cdfs_tst' not in TANGSHAN:
    import csv
    if isinstance(cdfs_tst, np.ndarray) or isinstance(cdfs_tst, pd.DataFrame
        ) or isinstance(cdfs_tst, pd.Series):
        shape_size = cdfs_tst.shape
    elif isinstance(cdfs_tst, list):
        shape_size = len(cdfs_tst)
    else:
        shape_size = 'any'
    check_type = type(cdfs_tst)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cdfs_tst')
        writer = csv.writer(f)
        writer.writerow(['cdfs_tst', 120, check_type, shape_size])
preds_full = create_full_predictions(cdfs_tst, preds)
if 'preds_full' not in TANGSHAN:
    import csv
    if isinstance(preds_full, np.ndarray) or isinstance(preds_full, pd.
        DataFrame) or isinstance(preds_full, pd.Series):
        shape_size = preds_full.shape
    elif isinstance(preds_full, list):
        shape_size = len(preds_full)
    else:
        shape_size = 'any'
    check_type = type(preds_full)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('preds_full')
        writer = csv.writer(f)
        writer.writerow(['preds_full', 121, check_type, shape_size])
labels = labels.reindex(integer_labels.index)
if 'integer_labels' not in TANGSHAN:
    import csv
    if isinstance(integer_labels, np.ndarray) or isinstance(integer_labels,
        pd.DataFrame) or isinstance(integer_labels, pd.Series):
        shape_size = integer_labels.shape
    elif isinstance(integer_labels, list):
        shape_size = len(integer_labels)
    else:
        shape_size = 'any'
    check_type = type(integer_labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('integer_labels')
        writer = csv.writer(f)
        writer.writerow(['integer_labels', 123, check_type, shape_size])
print(
    'my best score with 50% validation for this subset was 0.014, with a 430 features'
    )
print('CRPS using Reflectivity =', calc_crps(np.arange(70), preds_full,
    labels.iloc[:cutoff_value].values))
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
        writer.writerow(['print', 126, check_type, shape_size])
