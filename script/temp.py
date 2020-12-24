import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")
X2 = get_features(train_raw)
labels = np.array(train_raw['label'])
m = labels.shape[0]
y = np.zeros((m, 10))
for j in range(10):
    y[:, j] = (labels == j) * 1
folds = 5
oinst = 1
h_layers = 4
cv_groups = cross_validated(X2, folds)
iterations = 100
seeds = []
for j in range(oinst):
    batch_processing = True
    base_batch_size = 1024  # min size
    X = X2  # Direct Map
    n = [X.shape[1]]
    acts = ['input']
    for layer in range(h_layers):
        n.append((17) ** 2)
    L = len(n) - 1
    X_train = X[cv_groups[0][0], :].T
    y_train = y[cv_groups[0][0], :].T
    labels_train = labels[cv_groups[0][0]]
    depth = 1024
    filter1 = np.zeros((n[0], n[0]))
    for dim in range(10):
        for monomial in range(1, min(2, h_layers)):
            X_sample = X_train[:, :depth].T ** monomial
            X_mean = np.reshape(np.mean(X_sample, axis=0), (1, -1))
            y_sample = np.reshape(y_train[dim, :depth], (-1, 1))
            y_mean = np.mean(y_sample)
            y_var = (y_sample - y_mean) * X_sample ** 0
            numer = (np.dot((X_sample - X_mean).T, y_var))
            denom = np.sqrt(np.sum(np.dot((X_sample - X_mean).T, (X_sample - X_mean)))) * np.sqrt(
                np.dot((y_sample - y_mean).T, (y_sample - y_mean)))
            filter1 += np.abs(np.diag((numer / denom)[:, 0]))
    filter1 /= np.linalg.norm(filter1)
    filter2 = 1 * (np.abs(filter1) > 0.0001)