"""

This is not a model that brougth me to the 5th place

This one is based on post-competition forum chat re. ensemblig strong and weak classifiers, and

particularly on remarks made by Josef Slavicek (3rd place) who wonder why raising strong classifier's

prediction to a big power works so well, and on my thought what a physical soundness of a model is.

Should produce a score 0.999+ with n_models=5+ and n_epochs=100..120 (increase and run locally)

"""
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
import pandas as pd
import numpy as np
np.random.seed(1337)


def add_features(df):
    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP'] * df['dira']
    df['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']
    df['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, (['DOCAone', 'DOCAtwo', 'DOCAthree'])].max(axis=1
        )
    if 'df' not in TANGSHAN:
        import csv
        if isinstance(df, np.ndarray) or isinstance(df, pd.DataFrame
            ) or isinstance(df, pd.Series):
            shape_size = df.shape
        elif isinstance(df, list):
            shape_size = len(df)
        else:
            shape_size = 'any'
        check_type = type(df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('df')
            writer = csv.writer(f)
            writer.writerow(['df', 49, check_type, shape_size])
    df['iso_bdt_min'] = df.loc[:, (['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT'])
        ].min(axis=1)
    df['iso_min'] = df.loc[:, (['isolationa', 'isolationb', 'isolationc',
        'isolationd', 'isolatione', 'isolationf'])].min(axis=1)
    df['NEW_FD_SUMP'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] +
        df['p2_p'])
    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']
        ) / 3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, (['p0_track_Chi2Dof',
        'p1_track_Chi2Dof', 'p2_track_Chi2Dof'])].max(axis=1)
    df['NEW_FD_LT'] = df['FlightDistance'] / df['LifeTime']
    df['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']
        ) ** 2
    return df


def load_data(data_file, output_y=True):
    df = pd.read_csv(data_file)
    df = add_features(df)
    if output_y:
        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal'
        ]
    features = list(f for f in df.columns if f not in filter_out)
    return df[features].values, df['signal'].values if output_y else None, df[
        'id']


def model_factory(n_inputs):
    model = Sequential()
    model.add(Dense(n_inputs, 800))
    model.add(PReLU((800,)))
    model.add(Dropout(0.5))
    model.add(Dense(800, 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
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
            writer.writerow(['model', 107, check_type, shape_size])


X_train, y_train, _ = load_data('../input/training.csv')
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
        writer.writerow(['_', 113, check_type, shape_size])
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
        writer.writerow(['X_train', 113, check_type, shape_size])
X_test, _, id = load_data('../input/test.csv', output_y=False)
if 'id' not in TANGSHAN:
    import csv
    if isinstance(id, np.ndarray) or isinstance(id, pd.DataFrame
        ) or isinstance(id, pd.Series):
        shape_size = id.shape
    elif isinstance(id, list):
        shape_size = len(id)
    else:
        shape_size = 'any'
    check_type = type(id)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('id')
        writer = csv.writer(f)
        writer.writerow(['id', 115, check_type, shape_size])
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
y_train = np_utils.to_categorical(y_train)
if 'y_train' not in TANGSHAN:
    import csv
    if isinstance(y_train, np.ndarray) or isinstance(y_train, pd.DataFrame
        ) or isinstance(y_train, pd.Series):
        shape_size = y_train.shape
    elif isinstance(y_train, list):
        shape_size = len(y_train)
    else:
        shape_size = 'any'
    check_type = type(y_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_train')
        writer = csv.writer(f)
        writer.writerow(['y_train', 123, check_type, shape_size])
X_test = scaler.transform(X_test)
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
        writer.writerow(['scaler', 125, check_type, shape_size])
n_models = 1
if 'n_models' not in TANGSHAN:
    import csv
    if isinstance(n_models, np.ndarray) or isinstance(n_models, pd.DataFrame
        ) or isinstance(n_models, pd.Series):
        shape_size = n_models.shape
    elif isinstance(n_models, list):
        shape_size = len(n_models)
    else:
        shape_size = 'any'
    check_type = type(n_models)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('n_models')
        writer = csv.writer(f)
        writer.writerow(['n_models', 131, check_type, shape_size])
n_epochs = 100
probs = None
for i in range(n_models):
    print('\n----------- Keras: train Model %d/%d ----------\n' % (i + 1,
        n_models))
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
            writer.writerow(['print', 139, check_type, shape_size])
    model = model_factory(X_train.shape[1])
    model.fit(X_train, y_train, batch_size=64, nb_epoch=n_epochs,
        validation_data=None, verbose=2, show_accuracy=True)
    if 'n_epochs' not in TANGSHAN:
        import csv
        if isinstance(n_epochs, np.ndarray) or isinstance(n_epochs, pd.
            DataFrame) or isinstance(n_epochs, pd.Series):
            shape_size = n_epochs.shape
        elif isinstance(n_epochs, list):
            shape_size = len(n_epochs)
        else:
            shape_size = 'any'
        check_type = type(n_epochs)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('n_epochs')
            writer = csv.writer(f)
            writer.writerow(['n_epochs', 143, check_type, shape_size])
    p = model.predict(X_test, batch_size=256, verbose=0)[:, (1)]
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
            writer.writerow(['X_test', 145, check_type, shape_size])
    probs = p if probs is None else probs + p
    if 'probs' not in TANGSHAN:
        import csv
        if isinstance(probs, np.ndarray) or isinstance(probs, pd.DataFrame
            ) or isinstance(probs, pd.Series):
            shape_size = probs.shape
        elif isinstance(probs, list):
            shape_size = len(probs)
        else:
            shape_size = 'any'
        check_type = type(probs)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('probs')
            writer = csv.writer(f)
            writer.writerow(['probs', 147, check_type, shape_size])
    if 'p' not in TANGSHAN:
        import csv
        if isinstance(p, np.ndarray) or isinstance(p, pd.DataFrame
            ) or isinstance(p, pd.Series):
            shape_size = p.shape
        elif isinstance(p, list):
            shape_size = len(p)
        else:
            shape_size = 'any'
        check_type = type(p)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('p')
            writer = csv.writer(f)
            writer.writerow(['p', 147, check_type, shape_size])
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
        writer.writerow(['i', 137, check_type, shape_size])
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
        writer.writerow(['range', 137, check_type, shape_size])
probs /= n_models
np.random.seed(1337)
random_classifier = np.random.rand(len(probs))
q = 0.98
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
        writer.writerow(['q', 161, check_type, shape_size])
combined_probs = q * probs ** 30 + (1 - q) * random_classifier
if 'random_classifier' not in TANGSHAN:
    import csv
    if isinstance(random_classifier, np.ndarray) or isinstance(
        random_classifier, pd.DataFrame) or isinstance(random_classifier,
        pd.Series):
        shape_size = random_classifier.shape
    elif isinstance(random_classifier, list):
        shape_size = len(random_classifier)
    else:
        shape_size = 'any'
    check_type = type(random_classifier)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('random_classifier')
        writer = csv.writer(f)
        writer.writerow(['random_classifier', 163, check_type, shape_size])
df = pd.DataFrame({'id': id, 'prediction': combined_probs})
if 'combined_probs' not in TANGSHAN:
    import csv
    if isinstance(combined_probs, np.ndarray) or isinstance(combined_probs,
        pd.DataFrame) or isinstance(combined_probs, pd.Series):
        shape_size = combined_probs.shape
    elif isinstance(combined_probs, list):
        shape_size = len(combined_probs)
    else:
        shape_size = 'any'
    check_type = type(combined_probs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('combined_probs')
        writer = csv.writer(f)
        writer.writerow(['combined_probs', 165, check_type, shape_size])
df.to_csv('submission.csv', index=False)
