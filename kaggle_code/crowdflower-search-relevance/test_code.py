import numpy as np
import xgboost as xgb
from ml_metrics import quadratic_weighted_kappa


def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score - np.max(score))
    score /= np.sum(score, axis=1)[:, (np.newaxis)]
    return score


def softkappaobj(preds, dtrain):
    labels = dtrain.get_label() + 1
    labels = np.asarray(labels, dtype=int)
    preds = softmax(preds)
    M = preds.shape[0]
    N = preds.shape[1]
    O = 0.0
    for j in range(N):
        wj = (labels - (j + 1.0)) ** 2
        O += np.sum(wj * preds[:, (j)])
    hist_label = np.bincount(labels)[1:]
    hist_pred = np.sum(preds, axis=0)
    E = 0.0
    for i in range(N):
        for j in range(N):
            E += pow(i - j, 2.0) * hist_label[i] * hist_pred[j]
    grad = np.zeros((M, N))
    hess = np.zeros((M, N))
    for n in range(N):
        dO = np.zeros(M)
        for j in range(N):
            indicator = float(n == j)
            dO += (labels - (j + 1.0)) ** 2 * preds[:, (n)] * (indicator -
                preds[:, (j)])
        dE = np.zeros(M)
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                dE += pow(k - l, 2.0) * hist_label[l] * preds[:, (n)] * (
                    indicator - preds[:, (k)])
        grad[:, (n)] = -M * (dO * E - O * dE) / E ** 2
        d2O = np.zeros(M)
        for j in range(N):
            indicator = float(n == j)
            d2O += (labels - (j + 1.0)) ** 2 * preds[:, (n)] * (1 - 2.0 *
                preds[:, (n)]) * (indicator - preds[:, (j)])
        d2E = np.zeros(M)
        for k in range(N):
            for l in range(N):
                indicator = float(n == k)
                d2E += pow(k - l, 2.0) * hist_label[l] * preds[:, (n)] * (1 -
                    2.0 * preds[:, (n)]) * (indicator - preds[:, (k)])
        hess[:, (n)] = -M * ((d2O * E - O * d2E) * E ** 2 - (dO * E - O *
            dE) * 2.0 * E * dE) / E ** 4
    grad *= -1.0
    hess *= -1.0
    scale = 0.000125 / np.mean(abs(hess))
    hess *= scale
    hess = np.abs(hess)
    grad.shape = M * N
    hess.shape = M * N
    return grad, hess


def evalerror(preds, dtrain):
    labels = dtrain.get_label() + 1
    preds = softmax(preds)
    pred_labels = np.argmax(preds, axis=1) + 1
    kappa = quadratic_weighted_kappa(labels, pred_labels)
    return 'kappa', kappa


param = {'objective': 'reg:linear', 'num_class': 4}
param['booster'] = 'gblinear'
param['eta'] = 1
param['lambda'] = 5e-05
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
        writer.writerow(['param', 241, check_type, shape_size])
param['alpha'] = 1e-06
dtrain = xgb.DMatrix('path-to-train-feat', silent=True)
dvalid = xgb.DMatrix('path-to-valid-feat', silent=True)
num_round = 10
if 'num_round' not in TANGSHAN:
    import csv
    if isinstance(num_round, np.ndarray) or isinstance(num_round, pd.DataFrame
        ) or isinstance(num_round, pd.Series):
        shape_size = num_round.shape
    elif isinstance(num_round, list):
        shape_size = len(num_round)
    else:
        shape_size = 'any'
    check_type = type(num_round)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('num_round')
        writer = csv.writer(f)
        writer.writerow(['num_round', 257, check_type, shape_size])
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
if 'dvalid' not in TANGSHAN:
    import csv
    if isinstance(dvalid, np.ndarray) or isinstance(dvalid, pd.DataFrame
        ) or isinstance(dvalid, pd.Series):
        shape_size = dvalid.shape
    elif isinstance(dvalid, list):
        shape_size = len(dvalid)
    else:
        shape_size = 'any'
    check_type = type(dvalid)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dvalid')
        writer = csv.writer(f)
        writer.writerow(['dvalid', 259, check_type, shape_size])
bst = xgb.train(param, dtrain, num_round, watchlist, obj=softkappaobj,
    feval=evalerror)
if 'watchlist' not in TANGSHAN:
    import csv
    if isinstance(watchlist, np.ndarray) or isinstance(watchlist, pd.DataFrame
        ) or isinstance(watchlist, pd.Series):
        shape_size = watchlist.shape
    elif isinstance(watchlist, list):
        shape_size = len(watchlist)
    else:
        shape_size = 'any'
    check_type = type(watchlist)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('watchlist')
        writer = csv.writer(f)
        writer.writerow(['watchlist', 261, check_type, shape_size])
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
        writer.writerow(['dtrain', 261, check_type, shape_size])
pred = softmax(bst.predict(dvalid))
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
        writer.writerow(['bst', 267, check_type, shape_size])
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
        writer.writerow(['pred', 267, check_type, shape_size])
