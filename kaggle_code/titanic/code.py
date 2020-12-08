import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# print(train_data.head())

def clean_data(data, drop=True):
    data.Age.fillna(data.Age.mean(), inplace=True)
    data.Fare.fillna(data.Fare.mean(), inplace=True)

    if drop:
        data = data.dropna(axis=0)

    return data

def get_X(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # pound into shape
    pclass = data['Pclass'].values.reshape((-1, 1))
    male = (data['Sex'].values == 'male').astype(int).reshape((-1, 1))
    female = (data['Sex'].values == 'female').astype(int).reshape((-1, 1))
    sibsp = data['SibSp'].values.reshape((-1, 1))
    parch = data['Parch'].values.reshape((-1, 1))
    fare = data['Fare'].values.reshape((-1, 1))

    # normalize
    pclass = (pclass - pclass.mean()) / pclass.std()
    sibsp = (sibsp - sibsp.mean()) / sibsp.std()
    parch = (parch - parch.mean()) / parch.std()
    fare = (fare - fare.mean()) / fare.std()

    return np.concatenate((np.ones((data.shape[0], 1)), pclass, male, female, sibsp, parch, fare), axis=1)

def process_data(data):
    data = clean_data(data)

    y = data['Survived'].values.reshape((-1, 1))
    X = get_X(data)

    return X, y

def hypothesis(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(z))

def cost_function(X, y, theta):
    m = y.size
    h = hypothesis(X, theta)
    return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=0)

def gradient_descent(X, y, theta, alpha, iterations=1000):
    cost_history = []

    m = y.size

    for i in range(iterations):
        h = hypothesis(X, theta)
        delta = -np.dot((h - y).T, X).T / m # no idea why this is negative but it works

        cost_history.append(cost_function(X, y, theta))

        theta = theta - alpha * delta

    plt.plot(cost_history)
    plt.show()

    return theta

def predict(X, theta):
    return np.round(hypothesis(X, theta))

X, y = process_data(train_data)
theta = np.zeros((X.shape[1], 1))
print("Training...")

new_theta = gradient_descent(X, y, theta, 0.1, 1000)

# make predictions
print("Predicting")
test_X = get_X(clean_data(test_data, False))
predictions = predict(test_X, new_theta)

prediction_df = pd.DataFrame({'PassengerId': test_data.PassengerId.values, 'Survived': predictions.reshape((predictions.size))})

prediction_df.astype(int).to_csv('submission.csv', index=False, index_label=False)