from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold


'''
    This demonstrates how to run a Keras Deep Learning model for ROC AUC score (local 4-fold validation)
    for the springleaf challenge

    The model trains in a few seconds on CPU.
'''


def float32(k):
    return np.cast['float32'](k)
    
def load_train_data(path):
    print("Loading Train Data")
    df = pd.read_csv(path)
    
    
    # Remove line below to run locally - Be careful you need more than 8GB RAM 
    df = df.sample(n=80000)
    
    labels = df.target

    df = df.drop('target',1)
    df = df.drop('ID',1)
    
    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)

    X = df.values.copy()
    
    np.random.shuffle(X)

    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    print("Loading Test Data")
    df = pd.read_csv(path)
    ids = df.ID.astype(str)

    df = df.drop('ID',1)
    
    # Junk cols - Some feature engineering needed here
    df = df.ix[:, 520:660].fillna(-1)
    X = df.values.copy()

    X, = X.astype(np.float32),
    X = scaler.transform(X)
    return X, ids
    

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(input_dim, 32, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, 32, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, output_dim, init='lecun_uniform'))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer="adadelta")
    return model


if __name__ == "__main__":
    # Load data set and target values

    X, y, encoder, scaler = load_train_data("../input/train.csv")
    '''Convert class vector to binary class matrix, for use with categorical_crossentropy'''
    Y = np_utils.to_categorical(y)

    X_test, ids = load_test_data("../input/test.csv", scaler)
    print('Number of classes:', len(encoder.classes_))

    input_dim = X.shape[1]
    output_dim = len(encoder.classes_)

    print("Validation...")

    nb_folds = 4
    kfolds = KFold(len(y), nb_folds)
    av_roc = 0.
    f = 0
    for train, valid in kfolds:
        print('---'*20)
        print('Fold', f)
        print('---'*20)
        f += 1
        X_train = X[train]
        X_valid = X[valid]
        Y_train = Y[train]
        Y_valid = Y[valid]
        y_valid = y[valid]


        print("Building model...")
        model = build_model(input_dim, output_dim)

        print("Training model...")

        model.fit(X_train, Y_train, nb_epoch=10, batch_size=16, validation_data=(X_valid, Y_valid), verbose=0)
        valid_preds = model.predict_proba(X_valid, verbose=0)
        valid_preds = valid_preds[:, 1]
        roc = metrics.roc_auc_score(y_valid, valid_preds)
        print("ROC:", roc)
        av_roc += roc

    print('Average ROC:', av_roc/nb_folds)

    print("Generating submission...")

    model = build_model(input_dim, output_dim)
    model.fit(X, Y, nb_epoch=10, batch_size=16, verbose=0)

    preds = model.predict_proba(X_test, verbose=0)[:, 1]
    submission = pd.DataFrame(preds, index=ids, columns=['target'])
    submission.to_csv('Keras_BTB.csv')    
