# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def get_data():
    with open('../input/data.txt', 'r') as f:
        lines = f.readlines()
        T = int(lines[0])
        m, n = lines[1].split()
        m, n = int(m), int(n)
        data = np.zeros((m, n, T), dtype=int)
        for t, i in enumerate(range(2, T*m, m)):
            a = ''.join(lines[i:i+m])
            ak = np.fromstring(a, dtype=int, sep=' ').reshape((m, n))
            data[:, :, t] = ak
        return data

def write_result(data, result, fn_name):
    result = np.around(result).astype(int)
    unknowns = np.where(data == -1)
    output = open('./result_{}.csv'.format(fn_name), 'w')
    output.write('id,demand\n')
    for m_, n_, t_ in zip(*unknowns):
        output.write('{}:{}:{},{}\n'.format(t_, m_, n_, result[m_, n_, t_]))
    output.close()
    
def get_grid(m, n):
    return ((x, y) for x in range(m) for y in range(n))


def iter_rf(data, cont=True):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import pickle

    m, n, t = data.shape
    data_linear = np.zeros_like(data, dtype=np.float64)

    # 0 - linear interpolate data
    if cont:
        print("Loading prev data ...")
        inf = open('iter_rf.dat', 'rb')
        d = pickle.load(inf)
        data_inter = d['data_inter']
    else:
        # print("Linear interpolation missing data ...")
        # for m_, n_ in get_grid(m, n):
        #     a = data[m_, n_, :]
        #     x = np.where(a != -1)[0]
        #     y = a[x]

        #     data_linear[m_, n_, :] = np.interp(np.arange(t), x, y)

        # random imputation
        # data_inter = np.random.randn(m,n,t)

        for t_ in range(t):
            a = np.copy(data[:, :, t_])
            known = np.where(a != -1)
            unknown = np.where(a == -1)
            mean_t = np.mean(a[known])
            a[unknown] = mean_t
            data_linear[:, :, t_] = a

        data_inter = np.copy(data_linear)
        # print(data[:, :, 0])
        # print(data_inter[:, :, 0])

    for iter in range(3):
        print("===> Iter: ", iter)
        # 1 - create train/test data set
        data_index = np.where(data != -1)
        index_size = data_index[0].size
        feature_size = 1 + 1 + 2 + 64 + 2
        dataset = np.zeros((index_size, feature_size))
        Y = np.zeros((index_size, ))
        for i, k in enumerate(zip(*data_index)):
            m_, n_, t_ = k
            dt = data_inter[:, :, t_]
            pt, nt = 0, 0
            if t_ > 0:
                pt = data_inter[m_, n_, t_ - 1]
            if t_ < t - 1:
                nt = data_inter[m_, n_, t_ + 1]
            # dt = np.pad(dt, [(1, 1), (1, 1)], mode='constant')
            # dtr = dt[m_ :m_ + 3, n_:n_ + 3]
            tf = np.array([t_, t_ % 24, m_, n_, pt, nt])
            dataset[i, :] = np.concatenate([tf, dt.ravel()])
            Y[i] = data[m_, n_, t_]
        x_train, x_test, y_train, y_test = train_test_split(
            dataset, Y, test_size=0.2)
        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)

        # 2 - train random forest
        regr = RandomForestRegressor(n_estimators=75)
        # regr = MLPRegressor()
        # regr = SVR()
        regr.fit(x_train, y_train)
        y_pre = regr.predict(x_test)
        # print(y_pre.shape)
        error = np.sum(np.sqrt(np.power(y_test - y_pre, 2)))
        print("==> Error:", error)

        # 3 - fill data_inter with prediction
        pred_index = np.where(data == -1)
        pred_size = pred_index[0].size
        pred2in = {}
        dataset_pred = np.zeros((pred_size, feature_size))
        for i, k in enumerate(zip(*pred_index)):
            m_, n_, t_ = k
            dt = data_inter[:, :, t_]
            pt, nt = 0, 0
            if t_ > 0:
                pt = data_inter[m_, n_, t_ - 1]
            if t_ < t - 1:
                nt = data_inter[m_, n_, t_ + 1]
            # dt = np.pad(dt, [(1, 1), (1, 1)], mode='constant')
            # dtr = dt[m_:m_ + 3, n_ : n_ + 3]
            tf = np.array([t_, t_ % 24, m_, n_, pt, nt])
            dataset_pred[i, :] = np.concatenate([tf, dt.ravel()])
            pred2in[i] = k
        y_pre = regr.predict(dataset_pred)
        for i, y_val in enumerate(y_pre):
            m_, n_, t_ = pred2in[i]
            data_inter[m_, n_, t_] = y_val


    write_result(data, data_inter, 'iter_rf')

    outf = open('iter_rf.dat', 'wb')
    pickle.dump({'data_inter': data_inter}, outf)
    return


def plot_iter_rf(data):
    import pickle
    inf = open('iter_rf.dat', 'rb')
    d = pickle.load(inf)
    result = d['data_inter']

    from matplotlib import pyplot as plt
    for m, n in get_grid(8, 8):
        plt.plot(result[m, n, :])
        plt.waitforbuttonpress()
        plt.cla()


if __name__ == '__main__':
    data = get_data()
    iter_rf(data, cont=False)
    # plot_iter_rf(data)
