import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization


class NN:

    def __init__(self, inputShape, layers, dropout=[], activation='relu',
        init='uniform', loss='rmse', optimizer='adadelta', nb_epochs=50,
        batch_size=32, verbose=1):
        model = Sequential()
        for i in range(len(layers)):
            if i == 0:
                print('Input shape: ' + str(inputShape))
                print('Adding Layer ' + str(i) + ': ' + str(layers[i]))
                model.add(Dense(layers[i], input_dim=inputShape, init=init))
            else:
                print('Adding Layer ' + str(i) + ': ' + str(layers[i]))
                model.add(Dense(layers[i], init=init))
            print('Adding ' + activation + ' layer')
            model.add(Activation(activation))
            model.add(BatchNormalization())
            if len(dropout) > i:
                print('Adding ' + str(dropout[i]) + ' dropout')
                model.add(Dropout(dropout[i]))
        model.add(Dense(1, init=init))
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        self.model.fit(X.values, y.values, nb_epoch=self.nb_epochs,
            batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X, batch_size=128, verbose=1):
        return self.model.predict(X.values, batch_size=batch_size, verbose=
            verbose)


class pdStandardScaler:

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.StandardScaler = StandardScaler()

    def fit(self, df):
        self.StandardScaler.fit(df)

    def transform(self, df):
        df = pd.DataFrame(self.StandardScaler.transform(df), columns=df.columns
            )
        return df

    def fit_transform(self, df):
        df = pd.DataFrame(self.StandardScaler.fit_transform(df), columns=df
            .columns)
        return df


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


def pdFillNAN(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)


def make_dataset(useDummies=True, fillNANStrategy='mean', useNormalization=True
    ):
    data_dir = '../input/'
    train = pd.read_csv(data_dir + 'train.csv')
    test = pd.read_csv(data_dir + 'test.csv')
    labels = train['Response']
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
            writer.writerow(['labels', 193, check_type, shape_size])
    train.drop(labels='Id', axis=1, inplace=True)
    train.drop(labels='Response', axis=1, inplace=True)
    test.drop(labels='Id', axis=1, inplace=True)
    categoricalVariables = ['Product_Info_1', 'Product_Info_2',
        'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
        'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3',
        'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2',
        'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6',
        'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
        'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7',
        'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
        'Medical_History_2', 'Medical_History_3', 'Medical_History_4',
        'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
        'Medical_History_8', 'Medical_History_9', 'Medical_History_10',
        'Medical_History_11', 'Medical_History_12', 'Medical_History_13',
        'Medical_History_14', 'Medical_History_16', 'Medical_History_17',
        'Medical_History_18', 'Medical_History_19', 'Medical_History_20',
        'Medical_History_21', 'Medical_History_22', 'Medical_History_23',
        'Medical_History_25', 'Medical_History_26', 'Medical_History_27',
        'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
        'Medical_History_31', 'Medical_History_33', 'Medical_History_34',
        'Medical_History_35', 'Medical_History_36', 'Medical_History_37',
        'Medical_History_38', 'Medical_History_39', 'Medical_History_40',
        'Medical_History_41']
    if useDummies == True:
        print('Generating dummies...')
        train, test = getDummiesInplace(categoricalVariables, train, test)
    if fillNANStrategy is not None:
        print('Filling in missing values...')
        train = pdFillNAN(train, fillNANStrategy)
        test = pdFillNAN(test, fillNANStrategy)
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
                writer.writerow(['test', 221, check_type, shape_size])
    if useNormalization == True:
        print('Scaling...')
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
                writer.writerow(['print', 227, check_type, shape_size])
        scaler = pdStandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    return train, test, labels


print('Creating dataset...')
train, test, labels = make_dataset(useDummies=True, fillNANStrategy='mean',
    useNormalization=True)
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
        writer.writerow(['train', 243, check_type, shape_size])
clf = NN(inputShape=train.shape[1], layers=[128, 64], dropout=[0.5, 0.5],
    loss='mae', optimizer='adadelta', init='glorot_normal', nb_epochs=5)
print('Training model...')
clf.fit(train, labels)
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
        writer.writerow(['clf', 253, check_type, shape_size])
print('Making predictions...')
pred = clf.predict(test)
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
        writer.writerow(['pred', 259, check_type, shape_size])
predClipped = np.clip(np.round(pred), 1, 8).astype(int)
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
        writer.writerow(['int', 261, check_type, shape_size])
submission = pd.read_csv('../input/sample_submission.csv')
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
        writer.writerow(['submission', 265, check_type, shape_size])
submission['Response'] = predClipped
if 'predClipped' not in TANGSHAN:
    import csv
    if isinstance(predClipped, np.ndarray) or isinstance(predClipped, pd.
        DataFrame) or isinstance(predClipped, pd.Series):
        shape_size = predClipped.shape
    elif isinstance(predClipped, list):
        shape_size = len(predClipped)
    else:
        shape_size = 'any'
    check_type = type(predClipped)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('predClipped')
        writer = csv.writer(f)
        writer.writerow(['predClipped', 267, check_type, shape_size])
submission.to_csv('NNSubmission.csv', index=False)
