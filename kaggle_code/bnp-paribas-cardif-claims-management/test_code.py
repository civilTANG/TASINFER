"""

Created on Tue Feb 23 12:01:21 2016



@author: Ouranos

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid


class AdjustVariable(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


if 'object' not in TANGSHAN:
    import csv
    if isinstance(object, np.ndarray) or isinstance(object, pd.DataFrame
        ) or isinstance(object, pd.Series):
        shape_size = object.shape
    elif isinstance(object, list):
        shape_size = len(object)
    else:
        shape_size = 'any'
    check_type = type(object)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('object')
        writer = csv.writer(f)
        writer.writerow(['object', 43, check_type, shape_size])


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    if 'scaler' not in TANGSHAN:
        import csv
        if isinstance(scaler, np.ndarray) or isinstance(scaler, pd.DataFrame
            ) or isinstance(scaler, pd.Series):
            shape_size = scaler.shape
        elif isinstance(scaler, list):
            shape_size = len(scaler)
        else:
            shape_size = 'any'
        check_type = type(scaler)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('scaler')
            writer = csv.writer(f)
            writer.writerow(['scaler', 81, check_type, shape_size])
    return X, scaler


def getDummiesInplace(columnList, train, test=None):
    columns = []
    if test is not None:
        df = pd.concat([train, test], axis=0)
    else:
        df = train
    for columnName in df.columns:
        index = df.columns.get_loc(columnName)
        if columnName in columnList:
            dummies = pd.get_dummies(df.ix[:, (index)], prefix=columnName,
                prefix_sep='.')
            columns.append(dummies)
        else:
            columns.append(df.ix[:, (index)])
    df = pd.concat(columns, axis=1)
    if test is not None:
        train = df[:train.shape[0]]
        test = df[train.shape[0]:]
        return train, test
    else:
        train = df
        return train
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
            writer.writerow(['test', 129, check_type, shape_size])


def pdFillNAN(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]
labels = train['target']
trainId = train['ID']
if 'trainId' not in TANGSHAN:
    import csv
    if isinstance(trainId, np.ndarray) or isinstance(trainId, pd.DataFrame
        ) or isinstance(trainId, pd.Series):
        shape_size = trainId.shape
    elif isinstance(trainId, list):
        shape_size = len(trainId)
    else:
        shape_size = 'any'
    check_type = type(trainId)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('trainId')
        writer = csv.writer(f)
        writer.writerow(['trainId', 179, check_type, shape_size])
testId = test['ID']
if 'testId' not in TANGSHAN:
    import csv
    if isinstance(testId, np.ndarray) or isinstance(testId, pd.DataFrame
        ) or isinstance(testId, pd.Series):
        shape_size = testId.shape
    elif isinstance(testId, list):
        shape_size = len(testId)
    else:
        shape_size = 'any'
    check_type = type(testId)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('testId')
        writer = csv.writer(f)
        writer.writerow(['testId', 181, check_type, shape_size])
train.drop(['ID', 'target', 'v22', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37',
    'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81', 'v82',
    'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116',
    'v117', 'v118', 'v119', 'v123', 'v124', 'v128'], axis=1, inplace=True)
test.drop(labels=['ID', 'v22', 'v8', 'v23', 'v25', 'v31', 'v36', 'v37',
    'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v75', 'v79', 'v81', 'v82',
    'v89', 'v92', 'v95', 'v105', 'v107', 'v108', 'v109', 'v110', 'v116',
    'v117', 'v118', 'v119', 'v123', 'v124', 'v128'], axis=1, inplace=True)
categoricalVariables = []
for var in train.columns:
    vector = pd.concat([train[var], test[var]], axis=0)
    if 'vector' not in TANGSHAN:
        import csv
        if isinstance(vector, np.ndarray) or isinstance(vector, pd.DataFrame
            ) or isinstance(vector, pd.Series):
            shape_size = vector.shape
        elif isinstance(vector, list):
            shape_size = len(vector)
        else:
            shape_size = 'any'
        check_type = type(vector)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('vector')
            writer = csv.writer(f)
            writer.writerow(['vector', 203, check_type, shape_size])
    if 'var' not in TANGSHAN:
        import csv
        if isinstance(var, np.ndarray) or isinstance(var, pd.DataFrame
            ) or isinstance(var, pd.Series):
            shape_size = var.shape
        elif isinstance(var, list):
            shape_size = len(var)
        else:
            shape_size = 'any'
        check_type = type(var)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('var')
            writer = csv.writer(f)
            writer.writerow(['var', 203, check_type, shape_size])
    typ = str(train[var].dtype)
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
            writer.writerow(['str', 205, check_type, shape_size])
    if typ == 'object':
        categoricalVariables.append(var)
        if 'categoricalVariables' not in TANGSHAN:
            import csv
            if isinstance(categoricalVariables, np.ndarray) or isinstance(
                categoricalVariables, pd.DataFrame) or isinstance(
                categoricalVariables, pd.Series):
                shape_size = categoricalVariables.shape
            elif isinstance(categoricalVariables, list):
                shape_size = len(categoricalVariables)
            else:
                shape_size = 'any'
            check_type = type(categoricalVariables)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('categoricalVariables')
                writer = csv.writer(f)
                writer.writerow(['categoricalVariables', 209, check_type,
                    shape_size])
    if 'typ' not in TANGSHAN:
        import csv
        if isinstance(typ, np.ndarray) or isinstance(typ, pd.DataFrame
            ) or isinstance(typ, pd.Series):
            shape_size = typ.shape
        elif isinstance(typ, list):
            shape_size = len(typ)
        else:
            shape_size = 'any'
        check_type = type(typ)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('typ')
            writer = csv.writer(f)
            writer.writerow(['typ', 207, check_type, shape_size])
print('Generating dummies...')
train, test = getDummiesInplace(categoricalVariables, train, test)
cls = train.sum(axis=0)
train = train.drop(train.columns[cls < 10], axis=1)
test = test.drop(test.columns[cls < 10], axis=1)
if 'cls' not in TANGSHAN:
    import csv
    if isinstance(cls, np.ndarray) or isinstance(cls, pd.DataFrame
        ) or isinstance(cls, pd.Series):
        shape_size = cls.shape
    elif isinstance(cls, list):
        shape_size = len(cls)
    else:
        shape_size = 'any'
    check_type = type(cls)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('cls')
        writer = csv.writer(f)
        writer.writerow(['cls', 227, check_type, shape_size])
print('Filling in missing values...')
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
        writer.writerow(['print', 231, check_type, shape_size])
fillNANStrategy = -1
if 'fillNANStrategy' not in TANGSHAN:
    import csv
    if isinstance(fillNANStrategy, np.ndarray) or isinstance(fillNANStrategy,
        pd.DataFrame) or isinstance(fillNANStrategy, pd.Series):
        shape_size = fillNANStrategy.shape
    elif isinstance(fillNANStrategy, list):
        shape_size = len(fillNANStrategy)
    else:
        shape_size = 'any'
    check_type = type(fillNANStrategy)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('fillNANStrategy')
        writer = csv.writer(f)
        writer.writerow(['fillNANStrategy', 233, check_type, shape_size])
train = pdFillNAN(train, fillNANStrategy)
test = pdFillNAN(test, fillNANStrategy)
print('Scaling...')
train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)
train = np.asarray(train, dtype=np.float32)
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
        writer.writerow(['train', 255, check_type, shape_size])
labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
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
        writer.writerow(['labels', 257, check_type, shape_size])
net = NeuralNet(layers=[('input', InputLayer), ('dropout0', DropoutLayer),
    ('hidden1', DenseLayer), ('dropout1', DropoutLayer), ('hidden2',
    DenseLayer), ('output', DenseLayer)], input_shape=(None, len(train[1])),
    dropout0_p=0.1, hidden1_num_units=50, hidden1_W=Uniform(), dropout1_p=
    0.2, hidden2_num_units=40, output_nonlinearity=sigmoid,
    output_num_units=1, update=nesterov_momentum, update_learning_rate=
    theano.shared(np.float32(0.01)), update_momentum=theano.shared(np.
    float32(0.9)), on_epoch_finished=[AdjustVariable('update_learning_rate',
    start=0.01, stop=0.0001), AdjustVariable('update_momentum', start=0.9,
    stop=0.99)], regression=True, y_tensor_type=T.imatrix,
    objective_loss_function=binary_crossentropy, max_epochs=20, eval_size=
    0.1, verbose=2)
if 'net' not in TANGSHAN:
    import csv
    if isinstance(net, np.ndarray) or isinstance(net, pd.DataFrame
        ) or isinstance(net, pd.Series):
        shape_size = net.shape
    elif isinstance(net, list):
        shape_size = len(net)
    else:
        shape_size = 'any'
    check_type = type(net)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('net')
        writer = csv.writer(f)
        writer.writerow(['net', 261, check_type, shape_size])
seednumber = 1235
np.random.seed(seednumber)
if 'seednumber' not in TANGSHAN:
    import csv
    if isinstance(seednumber, np.ndarray) or isinstance(seednumber, pd.
        DataFrame) or isinstance(seednumber, pd.Series):
        shape_size = seednumber.shape
    elif isinstance(seednumber, list):
        shape_size = len(seednumber)
    else:
        shape_size = 'any'
    check_type = type(seednumber)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('seednumber')
        writer = csv.writer(f)
        writer.writerow(['seednumber', 339, check_type, shape_size])
net.fit(train, labels)
preds = net.predict_proba(test)[:, (0)]
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
        writer.writerow(['preds', 347, check_type, shape_size])
submission = pd.read_csv('../input/sample_submission.csv')
submission['PredictedProb'] = preds
submission.to_csv('NNbench.csv', index=False)
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
        writer.writerow(['submission', 357, check_type, shape_size])
