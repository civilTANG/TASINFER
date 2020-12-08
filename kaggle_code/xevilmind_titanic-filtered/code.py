# This Python 3 environment comes with many helpful anal,ytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras import backend as K
from keras import optimizers

# Read training and test data files
train = pd.read_csv("../input/titanic-filtered/numbers_train.csv").values
test  = pd.read_csv("../input/titanic-filtered/numbers_test.csv").values
test = test[:,1:]
train_x = train[:,2:]
print(train_x)

train_y = train[:,[1]]

def l1_reg(weight_matrix):
    return 0.04 * K.sum(K.abs(weight_matrix))

def create_model():
    """Create baseline model"""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=4))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model

def model_fit(model,X,Y):
    model.fit(X,Y, epochs=400, batch_size=16)
    pass

model = create_model()

X = train_x
Y = train_y

model = create_model()
model_fit(model,X,Y)

counter = 892

with open('output.csv','w') as file:
    file.write('PassengerId,Survived')
    file.write('\n')
    for x in test:
        result = model.predict_classes(np.array(x).reshape((1,4)))
        file.write(str(counter) + ',' + str(result[0][0]))
        file.write('\n')
        counter = counter + 1