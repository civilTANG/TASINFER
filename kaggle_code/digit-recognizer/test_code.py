import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from subprocess import check_output
TANGSHAN = []
print('Loading Raw Data')
train_raw = pd.read_csv('D:\dataset\digit-recognizer\\train.csv')
test_raw = pd.read_csv('D:\dataset\digit-recognizer\\test.csv')

def get_features(raw_data):
    cols = []
    for px in range(784):
        cols.append('pixel' + str(px))
    return (raw_data.as_matrix(cols) > 0) * 1


def cross_validated(X, n_samples):
    kf = KFold(n_samples, shuffle=True)
    result = [group for group in kf.split(X)]
    return result


def init_dnn_parameters(n, activations, epsilons, filter1=None):
    L = len(n)
    params = {}
    vgrad = {}
    d_rms = {}
    for l in range(1, L):
        W = np.random.randn(n[l], n[l - 1]) * epsilons[l]
        if 'epsilons' not in TANGSHAN:
            import csv
            if isinstance(epsilons, np.ndarray) or isinstance(epsilons, pd.
                DataFrame) or isinstance(epsilons, pd.Series):
                shape_size = epsilons.shape
            elif isinstance(epsilons, list):
                shape_size = len(epsilons)
            else:
                shape_size = 'any'
            check_type = type(epsilons)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('epsilons')
                writer = csv.writer(f)
                writer.writerow(['epsilons', 43, check_type, shape_size])
        if filter1 is not None and l == 1:
            W = np.dot(W, filter1)
        if 'filter1' not in TANGSHAN:
            import csv
            if isinstance(filter1, np.ndarray) or isinstance(filter1, pd.
                DataFrame) or isinstance(filter1, pd.Series):
                shape_size = filter1.shape
            elif isinstance(filter1, list):
                shape_size = len(filter1)
            else:
                shape_size = 'any'
            check_type = type(filter1)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('filter1')
                writer = csv.writer(f)
                writer.writerow(['filter1', 45, check_type, shape_size])
        b = np.zeros((n[l], 1))
        params['W' + str(l)] = W
        params['b' + str(l)] = b
        params['mu' + str(l)] = 0
        params['sig' + str(l)] = 1
        vgrad['W' + str(l)] = W * 0
        vgrad['b' + str(l)] = b * 0
        d_rms['W' + str(l)] = W * 0
        d_rms['b' + str(l)] = b * 0
        params['act' + str(l)] = activations[l]
    params['n'] = n
    if 'n' not in TANGSHAN:
        import csv
        if isinstance(n, np.ndarray) or isinstance(n, pd.DataFrame
            ) or isinstance(n, pd.Series):
            shape_size = n.shape
        elif isinstance(n, list):
            shape_size = len(n)
        else:
            shape_size = 'any'
        check_type = type(n)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('n')
            writer = csv.writer(f)
            writer.writerow(['n', 60, check_type, shape_size])
    return params, vgrad, d_rms


def gdnn(X, activation_function):
    leak_factor = 1 / 100000
    if activation_function == 'tanh':
        return np.tanh(X)
    if activation_function == 'lReLU':
        return (X > 0) * X + (X <= 0) * X * leak_factor
    if activation_function == 'linear':
        return X
    if activation_function == 'softmax':
        t = np.exp(X - np.max(X, axis=0))
        t_sum = np.reshape(np.sum(t, axis=0), (1, -1))
        return t / t_sum
    else:
        return 1 / (1 + np.exp(-X))
        if 'X' not in TANGSHAN:
            import csv
            if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame
                ) or isinstance(X, pd.Series):
                shape_size = X.shape
            elif isinstance(X, list):
                shape_size = len(X)
            else:
                shape_size = 'any'
            check_type = type(X)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('X')
                writer = csv.writer(f)
                writer.writerow(['X', 77, check_type, shape_size])


def gdnn_prime(X, activation_function):
    leak_factor = 1 / 100000
    if activation_function == 'tanh':
        return 1 - np.power(X, 2)
    if activation_function == 'lReLU':
        return (X > 0) * 1 + (X <= 0) * leak_factor
    if activation_function == 'linear':
        return X ** 0
    else:
        return 1 / (1 + np.exp(-X)) * (1 - 1 / (1 + np.exp(-X)))


def get_dnn_cost(Y_hat, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(Y_hat), Y)
    cost = -np.sum(logprobs) / m
    return cost
    if 'cost' not in TANGSHAN:
        import csv
        if isinstance(cost, np.ndarray) or isinstance(cost, pd.DataFrame
            ) or isinstance(cost, pd.Series):
            shape_size = cost.shape
        elif isinstance(cost, list):
            shape_size = len(cost)
        else:
            shape_size = 'any'
        check_type = type(cost)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('cost')
            writer = csv.writer(f)
            writer.writerow(['cost', 97, check_type, shape_size])


def forward_dnn_propagation(X, params):
    n = params['n']
    L = len(n)
    A_prev = X
    cache = {}
    cache['A' + str(0)] = X
    for l in range(1, L):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        Z = (np.dot(W, A_prev) + b - params['mu' + str(l)]) / (params['sig' +
            str(l)] + 1e-08)
        A = gdnn(Z, params['act' + str(l)])
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
        A_prev = A
    return A, cache, params
    if 'A' not in TANGSHAN:
        import csv
        if isinstance(A, np.ndarray) or isinstance(A, pd.DataFrame
            ) or isinstance(A, pd.Series):
            shape_size = A.shape
        elif isinstance(A, list):
            shape_size = len(A)
        else:
            shape_size = 'any'
        check_type = type(A)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('A')
            writer = csv.writer(f)
            writer.writerow(['A', 118, check_type, shape_size])


def back_dnn_propagation(X, Y, params, cache, alpha=0.01, _lambda=0,
    keep_prob=1):
    n = params['n']
    L = len(n) - 1
    m = X.shape[1]
    W_limit = 5
    A = cache['A' + str(L)]
    A1 = cache['A' + str(L - 1)]
    grads = {}
    dZ = A - Y
    dW = 1 / m * np.dot(dZ, A1.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    grads['dZ' + str(L)] = dZ
    grads['dW' + str(L)] = dW + _lambda / m * params['W' + str(L)]
    grads['db' + str(L)] = db
    params['W' + str(L)] -= alpha * dW
    params['b' + str(L)] -= alpha * db
    for l in reversed(range(1, L)):
        params['mu' + str(l)] = np.mean(cache['Z' + str(l)])
        params['sig' + str(l)] = np.std(cache['Z' + str(l)])
        dZ2 = dZ
        W2 = params['W' + str(l + 1)]
        b = params['b' + str(l)]
        A2 = cache['A' + str(l)]
        A1 = cache['A' + str(l - 1)]
        d = np.random.randn(A1.shape[0], A1.shape[1]) > keep_prob
        A1 = A1 * d / keep_prob
        dZ = np.dot(W2.T, dZ2) * gdnn_prime(A2, params['act' + str(l)])
        dW = 1 / m * np.dot(dZ, A1.T) + _lambda / m * params['W' + str(l)]
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        grads['dZ' + str(l)] = dZ
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        params['W' + str(l)] -= alpha * dW
        params['b' + str(l)] -= alpha * db
    return grads, params


def back_dnn_propagation_with_momentum(X, Y, params, cache, alpha=0.01,
    _lambda=0, keep_prob=1, beta=0.9, vgrad={}, d_rms={}, t=0):
    n = params['n']
    L = len(n) - 1
    beta2 = 0.999
    m = X.shape[1]
    W_limit = 5
    A = cache['A' + str(L)]
    A1 = cache['A' + str(L - 1)]
    grads = {}
    v_corr = {}
    s_corr = {}
    dZ = A - Y
    dW = 1 / m * np.dot(dZ, A1.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    grads['dZ' + str(L)] = dZ
    grads['dW' + str(L)] = dW + _lambda / m * params['W' + str(L)]
    if 'm' not in TANGSHAN:
        import csv
        if isinstance(m, np.ndarray) or isinstance(m, pd.DataFrame
            ) or isinstance(m, pd.Series):
            shape_size = m.shape
        elif isinstance(m, list):
            shape_size = len(m)
        else:
            shape_size = 'any'
        check_type = type(m)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('m')
            writer = csv.writer(f)
            writer.writerow(['m', 185, check_type, shape_size])
    grads['db' + str(L)] = db
    vgrad['W' + str(L)] = beta * vgrad['W' + str(L)] + (1 - beta) * grads[
        'dW' + str(L)]
    vgrad['b' + str(L)] = beta * vgrad['b' + str(L)] + (1 - beta) * grads[
        'db' + str(L)]
    v_corr['W' + str(L)] = vgrad['W' + str(L)] / (1 - beta ** t)
    v_corr['b' + str(L)] = vgrad['b' + str(L)] / (1 - beta ** t)
    if 'vgrad' not in TANGSHAN:
        import csv
        if isinstance(vgrad, np.ndarray) or isinstance(vgrad, pd.DataFrame
            ) or isinstance(vgrad, pd.Series):
            shape_size = vgrad.shape
        elif isinstance(vgrad, list):
            shape_size = len(vgrad)
        else:
            shape_size = 'any'
        check_type = type(vgrad)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('vgrad')
            writer = csv.writer(f)
            writer.writerow(['vgrad', 192, check_type, shape_size])
    d_rms['W' + str(L)] = beta2 * d_rms['W' + str(L)] + (1 - beta2) * grads[
        'dW' + str(L)] ** 2
    d_rms['b' + str(L)] = beta2 * d_rms['b' + str(L)] + (1 - beta2) * grads[
        'db' + str(L)] ** 2
    s_corr['W' + str(L)] = d_rms['W' + str(L)] / (1 - beta2 ** t)
    s_corr['b' + str(L)] = d_rms['b' + str(L)] / (1 - beta2 ** t)
    params['W' + str(L)] -= alpha * v_corr['W' + str(L)] / (np.sqrt(s_corr[
        'W' + str(L)]) + 1e-08)
    if 'L' not in TANGSHAN:
        import csv
        if isinstance(L, np.ndarray) or isinstance(L, pd.DataFrame
            ) or isinstance(L, pd.Series):
            shape_size = L.shape
        elif isinstance(L, list):
            shape_size = len(L)
        else:
            shape_size = 'any'
        check_type = type(L)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('L')
            writer = csv.writer(f)
            writer.writerow(['L', 204, check_type, shape_size])
    params['b' + str(L)] -= alpha * v_corr['b' + str(L)] / (np.sqrt(s_corr[
        'b' + str(L)]) + 1e-08)
    for l in reversed(range(1, L)):
        if l < L:
            params['mu' + str(l)] = (1 - alpha) * params['mu' + str(l)
                ] + alpha * np.reshape(np.nanmean(cache['Z' + str(l)], axis
                =1), (-1, 1))
            params['sig' + str(l)] = (1 - alpha) * params['sig' + str(l)
                ] + alpha * np.reshape(np.nanstd(cache['Z' + str(l)], axis=
                1), (-1, 1))
            if 'cache' not in TANGSHAN:
                import csv
                if isinstance(cache, np.ndarray) or isinstance(cache, pd.
                    DataFrame) or isinstance(cache, pd.Series):
                    shape_size = cache.shape
                elif isinstance(cache, list):
                    shape_size = len(cache)
                else:
                    shape_size = 'any'
                check_type = type(cache)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('cache')
                    writer = csv.writer(f)
                    writer.writerow(['cache', 210, check_type, shape_size])
        dZ2 = dZ
        W2 = params['W' + str(l + 1)]
        b = params['b' + str(l)]
        A2 = cache['A' + str(l)]
        A1 = cache['A' + str(l - 1)]
        d = np.random.randn(A1.shape[0], A1.shape[1]) > keep_prob
        A1 = A1 * d / keep_prob
        dZ = np.dot(W2.T, dZ2) * gdnn_prime(A2, params['act' + str(l)])
        dW = 1 / m * np.dot(dZ, A1.T) + _lambda / m * params['W' + str(l)]
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        grads['dZ' + str(l)] = dZ
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        vgrad['W' + str(l)] = beta * vgrad['W' + str(l)] + (1 - beta) * grads[
            'dW' + str(l)]
        vgrad['b' + str(l)] = beta * vgrad['b' + str(l)] + (1 - beta) * grads[
            'db' + str(l)]
        v_corr['W' + str(l)] = vgrad['W' + str(l)] / (1 - beta ** t)
        v_corr['b' + str(l)] = vgrad['b' + str(l)] / (1 - beta ** t)
        if 'beta' not in TANGSHAN:
            import csv
            if isinstance(beta, np.ndarray) or isinstance(beta, pd.DataFrame
                ) or isinstance(beta, pd.Series):
                shape_size = beta.shape
            elif isinstance(beta, list):
                shape_size = len(beta)
            else:
                shape_size = 'any'
            check_type = type(beta)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('beta')
                writer = csv.writer(f)
                writer.writerow(['beta', 228, check_type, shape_size])
        d_rms['W' + str(l)] = beta2 * d_rms['W' + str(l)] + (1 - beta2
            ) * grads['dW' + str(l)] ** 2
        d_rms['b' + str(l)] = beta2 * d_rms['b' + str(l)] + (1 - beta2
            ) * grads['db' + str(l)] ** 2
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
                writer.writerow(['str', 231, check_type, shape_size])
        s_corr['W' + str(l)] = d_rms['W' + str(l)] / (1 - beta2 ** t)
        if 'd_rms' not in TANGSHAN:
            import csv
            if isinstance(d_rms, np.ndarray) or isinstance(d_rms, pd.DataFrame
                ) or isinstance(d_rms, pd.Series):
                shape_size = d_rms.shape
            elif isinstance(d_rms, list):
                shape_size = len(d_rms)
            else:
                shape_size = 'any'
            check_type = type(d_rms)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('d_rms')
                writer = csv.writer(f)
                writer.writerow(['d_rms', 232, check_type, shape_size])
        s_corr['b' + str(l)] = d_rms['b' + str(l)] / (1 - beta2 ** t)
        params['W' + str(l)] -= alpha * v_corr['W' + str(l)] / (np.sqrt(
            s_corr['W' + str(l)]) + 1e-08)
        params['b' + str(l)] -= alpha * v_corr['b' + str(l)] / (np.sqrt(
            s_corr['b' + str(l)]) + 1e-08)
    return grads, params, vgrad, d_rms


def batch_back_propagation(X, Y, params, cache, alpha=0.01, _lambda=0,
    keep_prob=1, chunk_size=128, beta=0.9, vgrad={}, d_rms={}):
    m = X.shape[1]
    include_probability = keep_prob
    idx_from = 0
    batch_size = chunk_size
    idx_to = min(batch_size, m)
    print('Mini-Batch - Shuffling Training Data')
    shuffled_idx = list(np.random.permutation(m))
    X_shuffle = X[:, (shuffled_idx)]
    y_shuffle = Y[:, (shuffled_idx)]
    counter = 0
    while idx_to < m:
        counter += 1
        if idx_from < idx_to:
            X_train = X_shuffle[:, idx_from:idx_to]
            y_train = y_shuffle[:, idx_from:idx_to]
            A, cache, params = forward_dnn_propagation(X_train, params)
            grads, params, vgrad, d_rms = back_dnn_propagation_with_momentum(
                X_train, y_train, params, cache, alpha, _lambda, keep_prob,
                beta, vgrad, d_rms, counter)
        idx_from += batch_size
        idx_from = min(m, idx_from)
        idx_to += batch_size
        idx_to = min(m, idx_to)
    return grads, params, vgrad, d_rms
    if 'grads' not in TANGSHAN:
        import csv
        if isinstance(grads, np.ndarray) or isinstance(grads, pd.DataFrame
            ) or isinstance(grads, pd.Series):
            shape_size = grads.shape
        elif isinstance(grads, list):
            shape_size = len(grads)
        else:
            shape_size = 'any'
        check_type = type(grads)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('grads')
            writer = csv.writer(f)
            writer.writerow(['grads', 275, check_type, shape_size])


print('Loading Training and Dev Data ')
X2 = get_features(train_raw)
if 'X2' not in TANGSHAN:
    import csv
    if isinstance(X2, np.ndarray) or isinstance(X2, pd.DataFrame
        ) or isinstance(X2, pd.Series):
        shape_size = X2.shape
    elif isinstance(X2, list):
        shape_size = len(X2)
    else:
        shape_size = 'any'
    check_type = type(X2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X2')
        writer = csv.writer(f)
        writer.writerow(['X2', 279, check_type, shape_size])
if 'train_raw' not in TANGSHAN:
    import csv
    if isinstance(train_raw, np.ndarray) or isinstance(train_raw, pd.DataFrame
        ) or isinstance(train_raw, pd.Series):
        shape_size = train_raw.shape
    elif isinstance(train_raw, list):
        shape_size = len(train_raw)
    else:
        shape_size = 'any'
    check_type = type(train_raw)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_raw')
        writer = csv.writer(f)
        writer.writerow(['train_raw', 279, check_type, shape_size])
labels = np.array(train_raw['label'])
m = labels.shape[0]
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
        writer.writerow(['labels', 282, check_type, shape_size])
y = np.zeros((m, 10))
for j in range(10):
    y[:, (j)] = (labels == j) * 1
k = 38
if 'k' not in TANGSHAN:
    import csv
    if isinstance(k, np.ndarray) or isinstance(k, pd.DataFrame) or isinstance(k
        , pd.Series):
        shape_size = k.shape
    elif isinstance(k, list):
        shape_size = len(k)
    else:
        shape_size = 'any'
    check_type = type(k)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('k')
        writer = csv.writer(f)
        writer.writerow(['k', 287, check_type, shape_size])
folds = 5
oinst = 1
h_layers = 4
if 'h_layers' not in TANGSHAN:
    import csv
    if isinstance(h_layers, np.ndarray) or isinstance(h_layers, pd.DataFrame
        ) or isinstance(h_layers, pd.Series):
        shape_size = h_layers.shape
    elif isinstance(h_layers, list):
        shape_size = len(h_layers)
    else:
        shape_size = 'any'
    check_type = type(h_layers)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('h_layers')
        writer = csv.writer(f)
        writer.writerow(['h_layers', 290, check_type, shape_size])
beta = 0.9
np.random.seed(1)
print('Cross Validation using {} folds'.format(folds))
if 'folds' not in TANGSHAN:
    import csv
    if isinstance(folds, np.ndarray) or isinstance(folds, pd.DataFrame
        ) or isinstance(folds, pd.Series):
        shape_size = folds.shape
    elif isinstance(folds, list):
        shape_size = len(folds)
    else:
        shape_size = 'any'
    check_type = type(folds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('folds')
        writer = csv.writer(f)
        writer.writerow(['folds', 293, check_type, shape_size])
print('Building Deep Network of {} Hidden Layer Groups'.format(h_layers))
print('Cross Validation ..')
cv_groups = cross_validated(X2, folds)
print('Done')
alphas = np.linspace(0.00125, 0.00125, oinst)
epsilons = np.linspace(0.76, 0.78, oinst)
if 'oinst' not in TANGSHAN:
    import csv
    if isinstance(oinst, np.ndarray) or isinstance(oinst, pd.DataFrame
        ) or isinstance(oinst, pd.Series):
        shape_size = oinst.shape
    elif isinstance(oinst, list):
        shape_size = len(oinst)
    else:
        shape_size = 'any'
    check_type = type(oinst)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('oinst')
        writer = csv.writer(f)
        writer.writerow(['oinst', 299, check_type, shape_size])
gammas = np.linspace(0.01, 0.01, oinst)
if 'gammas' not in TANGSHAN:
    import csv
    if isinstance(gammas, np.ndarray) or isinstance(gammas, pd.DataFrame
        ) or isinstance(gammas, pd.Series):
        shape_size = gammas.shape
    elif isinstance(gammas, list):
        shape_size = len(gammas)
    else:
        shape_size = 'any'
    check_type = type(gammas)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('gammas')
        writer = csv.writer(f)
        writer.writerow(['gammas', 300, check_type, shape_size])
lambdas = np.linspace(1.0, 1.0, oinst)
if 'lambdas' not in TANGSHAN:
    import csv
    if isinstance(lambdas, np.ndarray) or isinstance(lambdas, pd.DataFrame
        ) or isinstance(lambdas, pd.Series):
        shape_size = lambdas.shape
    elif isinstance(lambdas, list):
        shape_size = len(lambdas)
    else:
        shape_size = 'any'
    check_type = type(lambdas)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('lambdas')
        writer = csv.writer(f)
        writer.writerow(['lambdas', 301, check_type, shape_size])
keep_probs = np.linspace(0.99, 0.99, oinst)
if 'keep_probs' not in TANGSHAN:
    import csv
    if isinstance(keep_probs, np.ndarray) or isinstance(keep_probs, pd.
        DataFrame) or isinstance(keep_probs, pd.Series):
        shape_size = keep_probs.shape
    elif isinstance(keep_probs, list):
        shape_size = len(keep_probs)
    else:
        shape_size = 'any'
    check_type = type(keep_probs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('keep_probs')
        writer = csv.writer(f)
        writer.writerow(['keep_probs', 302, check_type, shape_size])
alph_decays = np.linspace(0.9, 0.9, oinst)
if 'alph_decays' not in TANGSHAN:
    import csv
    if isinstance(alph_decays, np.ndarray) or isinstance(alph_decays, pd.
        DataFrame) or isinstance(alph_decays, pd.Series):
        shape_size = alph_decays.shape
    elif isinstance(alph_decays, list):
        shape_size = len(alph_decays)
    else:
        shape_size = 'any'
    check_type = type(alph_decays)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('alph_decays')
        writer = csv.writer(f)
        writer.writerow(['alph_decays', 303, check_type, shape_size])
iterations = 100
if 'iterations' not in TANGSHAN:
    import csv
    if isinstance(iterations, np.ndarray) or isinstance(iterations, pd.
        DataFrame) or isinstance(iterations, pd.Series):
        shape_size = iterations.shape
    elif isinstance(iterations, list):
        shape_size = len(iterations)
    else:
        shape_size = 'any'
    check_type = type(iterations)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('iterations')
        writer = csv.writer(f)
        writer.writerow(['iterations', 304, check_type, shape_size])
n_1 = []
break_tol = 1e-05
etscost = []
if 'etscost' not in TANGSHAN:
    import csv
    if isinstance(etscost, np.ndarray) or isinstance(etscost, pd.DataFrame
        ) or isinstance(etscost, pd.Series):
        shape_size = etscost.shape
    elif isinstance(etscost, list):
        shape_size = len(etscost)
    else:
        shape_size = 'any'
    check_type = type(etscost)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('etscost')
        writer = csv.writer(f)
        writer.writerow(['etscost', 307, check_type, shape_size])
etrcost = []
seeds = []
if 'seeds' not in TANGSHAN:
    import csv
    if isinstance(seeds, np.ndarray) or isinstance(seeds, pd.DataFrame
        ) or isinstance(seeds, pd.Series):
        shape_size = seeds.shape
    elif isinstance(seeds, list):
        shape_size = len(seeds)
    else:
        shape_size = 'any'
    check_type = type(seeds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('seeds')
        writer = csv.writer(f)
        writer.writerow(['seeds', 309, check_type, shape_size])
layers = []
for j in range(oinst):
    batch_processing = True
    base_batch_size = 1024
    print('Building Network')
    X = X2
    n = [X.shape[1]]
    acts = ['input']
    gamma = [0]
    for layer in range(h_layers):
        n.append(17 ** 2)
        acts.append('lReLU')
        gamma.append(np.sqrt(2 / n[layer - 1]))
        if 'gamma' not in TANGSHAN:
            import csv
            if isinstance(gamma, np.ndarray) or isinstance(gamma, pd.DataFrame
                ) or isinstance(gamma, pd.Series):
                shape_size = gamma.shape
            elif isinstance(gamma, list):
                shape_size = len(gamma)
            else:
                shape_size = 'any'
            check_type = type(gamma)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('gamma')
                writer = csv.writer(f)
                writer.writerow(['gamma', 323, check_type, shape_size])
    layers.append(j + 1)
    if 'layers' not in TANGSHAN:
        import csv
        if isinstance(layers, np.ndarray) or isinstance(layers, pd.DataFrame
            ) or isinstance(layers, pd.Series):
            shape_size = layers.shape
        elif isinstance(layers, list):
            shape_size = len(layers)
        else:
            shape_size = 'any'
        check_type = type(layers)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('layers')
            writer = csv.writer(f)
            writer.writerow(['layers', 330, check_type, shape_size])
    n.append(y.shape[1])
    acts.append('softmax')
    gamma.append(np.sqrt(1 / n[layer - 1]))
    if 'layer' not in TANGSHAN:
        import csv
        if isinstance(layer, np.ndarray) or isinstance(layer, pd.DataFrame
            ) or isinstance(layer, pd.Series):
            shape_size = layer.shape
        elif isinstance(layer, list):
            shape_size = len(layer)
        else:
            shape_size = 'any'
        check_type = type(layer)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('layer')
            writer = csv.writer(f)
            writer.writerow(['layer', 333, check_type, shape_size])
    n_1.append(j + 4)
    if 'n_1' not in TANGSHAN:
        import csv
        if isinstance(n_1, np.ndarray) or isinstance(n_1, pd.DataFrame
            ) or isinstance(n_1, pd.Series):
            shape_size = n_1.shape
        elif isinstance(n_1, list):
            shape_size = len(n_1)
        else:
            shape_size = 'any'
        check_type = type(n_1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('n_1')
            writer = csv.writer(f)
            writer.writerow(['n_1', 334, check_type, shape_size])
    np.random.seed(1)
    alpha = alphas[j]
    if 'alphas' not in TANGSHAN:
        import csv
        if isinstance(alphas, np.ndarray) or isinstance(alphas, pd.DataFrame
            ) or isinstance(alphas, pd.Series):
            shape_size = alphas.shape
        elif isinstance(alphas, list):
            shape_size = len(alphas)
        else:
            shape_size = 'any'
        check_type = type(alphas)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('alphas')
            writer = csv.writer(f)
            writer.writerow(['alphas', 337, check_type, shape_size])
    _lambda = lambdas[j]
    if 'j' not in TANGSHAN:
        import csv
        if isinstance(j, np.ndarray) or isinstance(j, pd.DataFrame
            ) or isinstance(j, pd.Series):
            shape_size = j.shape
        elif isinstance(j, list):
            shape_size = len(j)
        else:
            shape_size = 'any'
        check_type = type(j)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('j')
            writer = csv.writer(f)
            writer.writerow(['j', 338, check_type, shape_size])
    keep_prob = keep_probs[j]
    if 'keep_prob' not in TANGSHAN:
        import csv
        if isinstance(keep_prob, np.ndarray) or isinstance(keep_prob, pd.
            DataFrame) or isinstance(keep_prob, pd.Series):
            shape_size = keep_prob.shape
        elif isinstance(keep_prob, list):
            shape_size = len(keep_prob)
        else:
            shape_size = 'any'
        check_type = type(keep_prob)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('keep_prob')
            writer = csv.writer(f)
            writer.writerow(['keep_prob', 339, check_type, shape_size])
    epsilon = 0.76
    if 'epsilon' not in TANGSHAN:
        import csv
        if isinstance(epsilon, np.ndarray) or isinstance(epsilon, pd.DataFrame
            ) or isinstance(epsilon, pd.Series):
            shape_size = epsilon.shape
        elif isinstance(epsilon, list):
            shape_size = len(epsilon)
        else:
            shape_size = 'any'
        check_type = type(epsilon)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('epsilon')
            writer = csv.writer(f)
            writer.writerow(['epsilon', 340, check_type, shape_size])
    L = len(n) - 1
    X_train = X[(cv_groups[0][0]), :].T
    y_train = y[(cv_groups[0][0]), :].T
    labels_train = labels[cv_groups[0][0]]
    if 'labels_train' not in TANGSHAN:
        import csv
        if isinstance(labels_train, np.ndarray) or isinstance(labels_train,
            pd.DataFrame) or isinstance(labels_train, pd.Series):
            shape_size = labels_train.shape
        elif isinstance(labels_train, list):
            shape_size = len(labels_train)
        else:
            shape_size = 'any'
        check_type = type(labels_train)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('labels_train')
            writer = csv.writer(f)
            writer.writerow(['labels_train', 348, check_type, shape_size])
    depth = 1024
    print('Building Input Layer Initialization Filter, Depth = {}'.format(
        depth))
    filter1 = np.zeros((n[0], n[0]))
    for dim in range(10):
        for monomial in range(1, min(2, h_layers)):
            X_sample = X_train[:, :depth].T ** monomial
            if 'depth' not in TANGSHAN:
                import csv
                if isinstance(depth, np.ndarray) or isinstance(depth, pd.
                    DataFrame) or isinstance(depth, pd.Series):
                    shape_size = depth.shape
                elif isinstance(depth, list):
                    shape_size = len(depth)
                else:
                    shape_size = 'any'
                check_type = type(depth)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('depth')
                    writer = csv.writer(f)
                    writer.writerow(['depth', 356, check_type, shape_size])
            X_mean = np.reshape(np.mean(X_sample, axis=0), (1, -1))
            y_sample = np.reshape(y_train[(dim), :depth], (-1, 1))
            if 'dim' not in TANGSHAN:
                import csv
                if isinstance(dim, np.ndarray) or isinstance(dim, pd.DataFrame
                    ) or isinstance(dim, pd.Series):
                    shape_size = dim.shape
                elif isinstance(dim, list):
                    shape_size = len(dim)
                else:
                    shape_size = 'any'
                check_type = type(dim)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('dim')
                    writer = csv.writer(f)
                    writer.writerow(['dim', 358, check_type, shape_size])
            y_mean = np.mean(y_sample)
            y_var = (y_sample - y_mean) * X_sample ** 0
            if 'X_sample' not in TANGSHAN:
                import csv
                if isinstance(X_sample, np.ndarray) or isinstance(X_sample,
                    pd.DataFrame) or isinstance(X_sample, pd.Series):
                    shape_size = X_sample.shape
                elif isinstance(X_sample, list):
                    shape_size = len(X_sample)
                else:
                    shape_size = 'any'
                check_type = type(X_sample)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('X_sample')
                    writer = csv.writer(f)
                    writer.writerow(['X_sample', 361, check_type, shape_size])
            numer = np.dot((X_sample - X_mean).T, y_var)
            if 'y_var' not in TANGSHAN:
                import csv
                if isinstance(y_var, np.ndarray) or isinstance(y_var, pd.
                    DataFrame) or isinstance(y_var, pd.Series):
                    shape_size = y_var.shape
                elif isinstance(y_var, list):
                    shape_size = len(y_var)
                else:
                    shape_size = 'any'
                check_type = type(y_var)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('y_var')
                    writer = csv.writer(f)
                    writer.writerow(['y_var', 362, check_type, shape_size])
            denom = np.sqrt(np.sum(np.dot((X_sample - X_mean).T, X_sample -
                X_mean))) * np.sqrt(np.dot((y_sample - y_mean).T, y_sample -
                y_mean))
            if 'X_mean' not in TANGSHAN:
                import csv
                if isinstance(X_mean, np.ndarray) or isinstance(X_mean, pd.
                    DataFrame) or isinstance(X_mean, pd.Series):
                    shape_size = X_mean.shape
                elif isinstance(X_mean, list):
                    shape_size = len(X_mean)
                else:
                    shape_size = 'any'
                check_type = type(X_mean)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('X_mean')
                    writer = csv.writer(f)
                    writer.writerow(['X_mean', 363, check_type, shape_size])
            if 'y_sample' not in TANGSHAN:
                import csv
                if isinstance(y_sample, np.ndarray) or isinstance(y_sample,
                    pd.DataFrame) or isinstance(y_sample, pd.Series):
                    shape_size = y_sample.shape
                elif isinstance(y_sample, list):
                    shape_size = len(y_sample)
                else:
                    shape_size = 'any'
                check_type = type(y_sample)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('y_sample')
                    writer = csv.writer(f)
                    writer.writerow(['y_sample', 363, check_type, shape_size])
            if 'y_mean' not in TANGSHAN:
                import csv
                if isinstance(y_mean, np.ndarray) or isinstance(y_mean, pd.
                    DataFrame) or isinstance(y_mean, pd.Series):
                    shape_size = y_mean.shape
                elif isinstance(y_mean, list):
                    shape_size = len(y_mean)
                else:
                    shape_size = 'any'
                check_type = type(y_mean)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('y_mean')
                    writer = csv.writer(f)
                    writer.writerow(['y_mean', 363, check_type, shape_size])
            filter1 += np.abs(np.diag((numer / denom)[:, (0)]))
            if 'numer' not in TANGSHAN:
                import csv
                if isinstance(numer, np.ndarray) or isinstance(numer, pd.
                    DataFrame) or isinstance(numer, pd.Series):
                    shape_size = numer.shape
                elif isinstance(numer, list):
                    shape_size = len(numer)
                else:
                    shape_size = 'any'
                check_type = type(numer)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('numer')
                    writer = csv.writer(f)
                    writer.writerow(['numer', 364, check_type, shape_size])
            if 'denom' not in TANGSHAN:
                import csv
                if isinstance(denom, np.ndarray) or isinstance(denom, pd.
                    DataFrame) or isinstance(denom, pd.Series):
                    shape_size = denom.shape
                elif isinstance(denom, list):
                    shape_size = len(denom)
                else:
                    shape_size = 'any'
                check_type = type(denom)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('denom')
                    writer = csv.writer(f)
                    writer.writerow(['denom', 364, check_type, shape_size])
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
                writer.writerow(['range', 355, check_type, shape_size])
        if 'min' not in TANGSHAN:
            import csv
            if isinstance(min, np.ndarray) or isinstance(min, pd.DataFrame
                ) or isinstance(min, pd.Series):
                shape_size = min.shape
            elif isinstance(min, list):
                shape_size = len(min)
            else:
                shape_size = 'any'
            check_type = type(min)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('min')
                writer = csv.writer(f)
                writer.writerow(['min', 355, check_type, shape_size])
        if 'monomial' not in TANGSHAN:
            import csv
            if isinstance(monomial, np.ndarray) or isinstance(monomial, pd.
                DataFrame) or isinstance(monomial, pd.Series):
                shape_size = monomial.shape
            elif isinstance(monomial, list):
                shape_size = len(monomial)
            else:
                shape_size = 'any'
            check_type = type(monomial)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('monomial')
                writer = csv.writer(f)
                writer.writerow(['monomial', 355, check_type, shape_size])
    filter1 /= np.linalg.norm(filter1)
    filter2 = 1 * (np.abs(filter1) > 0.0001)
    if 'filter2' not in TANGSHAN:
        import csv
        if isinstance(filter2, np.ndarray) or isinstance(filter2, pd.DataFrame
            ) or isinstance(filter2, pd.Series):
            shape_size = filter2.shape
        elif isinstance(filter2, list):
            shape_size = len(filter2)
        else:
            shape_size = 'any'
        check_type = type(filter2)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('filter2')
            writer = csv.writer(f)
            writer.writerow(['filter2', 366, check_type, shape_size])
    params, vgrad, d_rms = init_dnn_parameters(n, acts, gamma)
    X_test = X[(cv_groups[0][1]), :].T
    if 'cv_groups' not in TANGSHAN:
        import csv
        if isinstance(cv_groups, np.ndarray) or isinstance(cv_groups, pd.
            DataFrame) or isinstance(cv_groups, pd.Series):
            shape_size = cv_groups.shape
        elif isinstance(cv_groups, list):
            shape_size = len(cv_groups)
        else:
            shape_size = 'any'
        check_type = type(cv_groups)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('cv_groups')
            writer = csv.writer(f)
            writer.writerow(['cv_groups', 371, check_type, shape_size])
    y_test = y[(cv_groups[0][1]), :].T
    if 'y' not in TANGSHAN:
        import csv
        if isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame
            ) or isinstance(y, pd.Series):
            shape_size = y.shape
        elif isinstance(y, list):
            shape_size = len(y)
        else:
            shape_size = 'any'
        check_type = type(y)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y')
            writer = csv.writer(f)
            writer.writerow(['y', 372, check_type, shape_size])
    print(
        'Experiment [{}] - Eps = {}, Alph = {:3.2f}, Decay = {:3.2f}, lambda={:3.2f}'
        .format(j, epsilon, alpha, alph_decays[j], _lambda))
    print('k = {}, |X| = {}, max(i) = {}'.format(k, X_test.shape[0],
        iterations))
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
            writer.writerow(['X_test', 374, check_type, shape_size])
    print('Network Size {}'.format(n))
    print('Network Activation{}'.format(acts))
    if 'acts' not in TANGSHAN:
        import csv
        if isinstance(acts, np.ndarray) or isinstance(acts, pd.DataFrame
            ) or isinstance(acts, pd.Series):
            shape_size = acts.shape
        elif isinstance(acts, list):
            shape_size = len(acts)
        else:
            shape_size = 'any'
        check_type = type(acts)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('acts')
            writer = csv.writer(f)
            writer.writerow(['acts', 377, check_type, shape_size])
    cost = []
    tcost = []
    print('Mini-Batch : [{}], Mini-Batch Size [{}]'.format(batch_processing,
        base_batch_size))
    if 'base_batch_size' not in TANGSHAN:
        import csv
        if isinstance(base_batch_size, np.ndarray) or isinstance(
            base_batch_size, pd.DataFrame) or isinstance(base_batch_size,
            pd.Series):
            shape_size = base_batch_size.shape
        elif isinstance(base_batch_size, list):
            shape_size = len(base_batch_size)
        else:
            shape_size = 'any'
        check_type = type(base_batch_size)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('base_batch_size')
            writer = csv.writer(f)
            writer.writerow(['base_batch_size', 380, check_type, shape_size])
    print('Measuring Cost for [Training Set]', end='')
    A, cache, params = forward_dnn_propagation(X_train, params)
    cost.append(np.mean(get_dnn_cost(A, y_train)))
    print(',[Dev. Set]')
    A2, vectors, _ = forward_dnn_propagation(X_test, params)
    tcost.append(get_dnn_cost(A2, y_test))
    print('Pre-Training Cost')
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
            writer.writerow(['print', 387, check_type, shape_size])
    print('i = {:3d}, trc = {:3.2f}, tsc={:3.2f}'.format(-1, cost[-1],
        tcost[-1]))
    print(' active alpha = {:.2E}'.format(alpha))
    for i in range(iterations):
        if batch_processing:
            batch_size = base_batch_size
            grads, params, vgrad, d_rms = batch_back_propagation(X_train,
                y_train, params, cache, alpha * (batch_size / 2048),
                _lambda, keep_prob, batch_size, beta ** (batch_size / 2048),
                vgrad, d_rms)
            if 'X_train' not in TANGSHAN:
                import csv
                if isinstance(X_train, np.ndarray) or isinstance(X_train,
                    pd.DataFrame) or isinstance(X_train, pd.Series):
                    shape_size = X_train.shape
                elif isinstance(X_train, list):
                    shape_size = len(X_train)
                else:
                    shape_size = 'any'
                check_type = type(X_train)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('X_train')
                    writer = csv.writer(f)
                    writer.writerow(['X_train', 396, check_type, shape_size])
            if '_lambda' not in TANGSHAN:
                import csv
                if isinstance(_lambda, np.ndarray) or isinstance(_lambda,
                    pd.DataFrame) or isinstance(_lambda, pd.Series):
                    shape_size = _lambda.shape
                elif isinstance(_lambda, list):
                    shape_size = len(_lambda)
                else:
                    shape_size = 'any'
                check_type = type(_lambda)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('_lambda')
                    writer = csv.writer(f)
                    writer.writerow(['_lambda', 396, check_type, shape_size])
            if 'params' not in TANGSHAN:
                import csv
                if isinstance(params, np.ndarray) or isinstance(params, pd.
                    DataFrame) or isinstance(params, pd.Series):
                    shape_size = params.shape
                elif isinstance(params, list):
                    shape_size = len(params)
                else:
                    shape_size = 'any'
                check_type = type(params)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('params')
                    writer = csv.writer(f)
                    writer.writerow(['params', 396, check_type, shape_size])
            if 'batch_size' not in TANGSHAN:
                import csv
                if isinstance(batch_size, np.ndarray) or isinstance(batch_size,
                    pd.DataFrame) or isinstance(batch_size, pd.Series):
                    shape_size = batch_size.shape
                elif isinstance(batch_size, list):
                    shape_size = len(batch_size)
                else:
                    shape_size = 'any'
                check_type = type(batch_size)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('batch_size')
                    writer = csv.writer(f)
                    writer.writerow(['batch_size', 396, check_type, shape_size]
                        )
            print('Epoch [{}], Evaluating, [Training] '.format(i), end='')
            A, cache, params = forward_dnn_propagation(X_train, params)
            cost.append(np.mean(get_dnn_cost(A, y_train)))
            print(' Evaluating, [Dev] ')
            A2, vectors, _ = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
        else:
            A, cache = forward_dnn_propagation(X_train, params)
            cost.append(get_dnn_cost(A, y_train))
            grads, params = back_dnn_propagation(X_train, y_train, params,
                cache, alpha, _lambda, keep_prob)
            if 'y_train' not in TANGSHAN:
                import csv
                if isinstance(y_train, np.ndarray) or isinstance(y_train,
                    pd.DataFrame) or isinstance(y_train, pd.Series):
                    shape_size = y_train.shape
                elif isinstance(y_train, list):
                    shape_size = len(y_train)
                else:
                    shape_size = 'any'
                check_type = type(y_train)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('y_train')
                    writer = csv.writer(f)
                    writer.writerow(['y_train', 407, check_type, shape_size])
            A2, vectors = forward_dnn_propagation(X_test, params)
            tcost.append(get_dnn_cost(A2, y_test))
            if 'tcost' not in TANGSHAN:
                import csv
                if isinstance(tcost, np.ndarray) or isinstance(tcost, pd.
                    DataFrame) or isinstance(tcost, pd.Series):
                    shape_size = tcost.shape
                elif isinstance(tcost, list):
                    shape_size = len(tcost)
                else:
                    shape_size = 'any'
                check_type = type(tcost)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('tcost')
                    writer = csv.writer(f)
                    writer.writerow(['tcost', 413, check_type, shape_size])
        if 'batch_processing' not in TANGSHAN:
            import csv
            if isinstance(batch_processing, np.ndarray) or isinstance(
                batch_processing, pd.DataFrame) or isinstance(batch_processing,
                pd.Series):
                shape_size = batch_processing.shape
            elif isinstance(batch_processing, list):
                shape_size = len(batch_processing)
            else:
                shape_size = 'any'
            check_type = type(batch_processing)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('batch_processing')
                writer = csv.writer(f)
                writer.writerow(['batch_processing', 392, check_type,
                    shape_size])
        if alpha * np.abs(np.linalg.norm(grads['dW' + str(L)])) < break_tol:
            print('Reached Change Tolerance')
            break
        if 'break_tol' not in TANGSHAN:
            import csv
            if isinstance(break_tol, np.ndarray) or isinstance(break_tol,
                pd.DataFrame) or isinstance(break_tol, pd.Series):
                shape_size = break_tol.shape
            elif isinstance(break_tol, list):
                shape_size = len(break_tol)
            else:
                shape_size = 'any'
            check_type = type(break_tol)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('break_tol')
                writer = csv.writer(f)
                writer.writerow(['break_tol', 415, check_type, shape_size])
        if i % 1 == 0:
            alpha *= 1 - alpha
            print(' active alpha = {:.2E}'.format(alpha))
            if 'alpha' not in TANGSHAN:
                import csv
                if isinstance(alpha, np.ndarray) or isinstance(alpha, pd.
                    DataFrame) or isinstance(alpha, pd.Series):
                    shape_size = alpha.shape
                elif isinstance(alpha, list):
                    shape_size = len(alpha)
                else:
                    shape_size = 'any'
                check_type = type(alpha)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('alpha')
                    writer = csv.writer(f)
                    writer.writerow(['alpha', 420, check_type, shape_size])
            if 1 == 1:
                print('Number Matching (Dev. Set)')
                for num in range(10):
                    y_hat = A2[(num), :] > 0.5
                    if 'num' not in TANGSHAN:
                        import csv
                        if isinstance(num, np.ndarray) or isinstance(num,
                            pd.DataFrame) or isinstance(num, pd.Series):
                            shape_size = num.shape
                        elif isinstance(num, list):
                            shape_size = len(num)
                        else:
                            shape_size = 'any'
                        check_type = type(num)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('num')
                            writer = csv.writer(f)
                            writer.writerow(['num', 424, check_type,
                                shape_size])
                    y_star = y_test[(num), :]
                    if 'y_test' not in TANGSHAN:
                        import csv
                        if isinstance(y_test, np.ndarray) or isinstance(y_test,
                            pd.DataFrame) or isinstance(y_test, pd.Series):
                            shape_size = y_test.shape
                        elif isinstance(y_test, list):
                            shape_size = len(y_test)
                        else:
                            shape_size = 'any'
                        check_type = type(y_test)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('y_test')
                            writer = csv.writer(f)
                            writer.writerow(['y_test', 425, check_type,
                                shape_size])
                    matched = np.sum((1 - np.abs(y_star - y_hat)) * y_star)
                    if 'matched' not in TANGSHAN:
                        import csv
                        if isinstance(matched, np.ndarray) or isinstance(
                            matched, pd.DataFrame) or isinstance(matched,
                            pd.Series):
                            shape_size = matched.shape
                        elif isinstance(matched, list):
                            shape_size = len(matched)
                        else:
                            shape_size = 'any'
                        check_type = type(matched)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('matched')
                            writer = csv.writer(f)
                            writer.writerow(['matched', 426, check_type,
                                shape_size])
                    tp = np.sum((y_hat == y_star) * y_star * 1)
                    if 'tp' not in TANGSHAN:
                        import csv
                        if isinstance(tp, np.ndarray) or isinstance(tp, pd.
                            DataFrame) or isinstance(tp, pd.Series):
                            shape_size = tp.shape
                        elif isinstance(tp, list):
                            shape_size = len(tp)
                        else:
                            shape_size = 'any'
                        check_type = type(tp)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('tp')
                            writer = csv.writer(f)
                            writer.writerow(['tp', 427, check_type, shape_size]
                                )
                    if 'y_star' not in TANGSHAN:
                        import csv
                        if isinstance(y_star, np.ndarray) or isinstance(y_star,
                            pd.DataFrame) or isinstance(y_star, pd.Series):
                            shape_size = y_star.shape
                        elif isinstance(y_star, list):
                            shape_size = len(y_star)
                        else:
                            shape_size = 'any'
                        check_type = type(y_star)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('y_star')
                            writer = csv.writer(f)
                            writer.writerow(['y_star', 427, check_type,
                                shape_size])
                    if 'y_hat' not in TANGSHAN:
                        import csv
                        if isinstance(y_hat, np.ndarray) or isinstance(y_hat,
                            pd.DataFrame) or isinstance(y_hat, pd.Series):
                            shape_size = y_hat.shape
                        elif isinstance(y_hat, list):
                            shape_size = len(y_hat)
                        else:
                            shape_size = 'any'
                        check_type = type(y_hat)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('y_hat')
                            writer = csv.writer(f)
                            writer.writerow(['y_hat', 427, check_type,
                                shape_size])
                    tn = np.sum((y_hat == y_star) * (1 - y_star) * 1)
                    if 'tn' not in TANGSHAN:
                        import csv
                        if isinstance(tn, np.ndarray) or isinstance(tn, pd.
                            DataFrame) or isinstance(tn, pd.Series):
                            shape_size = tn.shape
                        elif isinstance(tn, list):
                            shape_size = len(tn)
                        else:
                            shape_size = 'any'
                        check_type = type(tn)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('tn')
                            writer = csv.writer(f)
                            writer.writerow(['tn', 428, check_type, shape_size]
                                )
                    fp = np.sum((y_hat == 1 - y_star) * (1 - y_star) * 1)
                    if 'fp' not in TANGSHAN:
                        import csv
                        if isinstance(fp, np.ndarray) or isinstance(fp, pd.
                            DataFrame) or isinstance(fp, pd.Series):
                            shape_size = fp.shape
                        elif isinstance(fp, list):
                            shape_size = len(fp)
                        else:
                            shape_size = 'any'
                        check_type = type(fp)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('fp')
                            writer = csv.writer(f)
                            writer.writerow(['fp', 429, check_type, shape_size]
                                )
                    fn = np.sum((y_hat == 1 - y_star) * y_star * 1)
                    if 'fn' not in TANGSHAN:
                        import csv
                        if isinstance(fn, np.ndarray) or isinstance(fn, pd.
                            DataFrame) or isinstance(fn, pd.Series):
                            shape_size = fn.shape
                        elif isinstance(fn, list):
                            shape_size = len(fn)
                        else:
                            shape_size = 'any'
                        check_type = type(fn)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('fn')
                            writer = csv.writer(f)
                            writer.writerow(['fn', 430, check_type, shape_size]
                                )
                    distance = np.linalg.norm((y_star - A2[(num), :]) * y_star)
                    if 'distance' not in TANGSHAN:
                        import csv
                        if isinstance(distance, np.ndarray) or isinstance(
                            distance, pd.DataFrame) or isinstance(distance,
                            pd.Series):
                            shape_size = distance.shape
                        elif isinstance(distance, list):
                            shape_size = len(distance)
                        else:
                            shape_size = 'any'
                        check_type = type(distance)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('distance')
                            writer = csv.writer(f)
                            writer.writerow(['distance', 431, check_type,
                                shape_size])
                    m_test = sum(y_test[(num), :] == 1)
                    if 'm_test' not in TANGSHAN:
                        import csv
                        if isinstance(m_test, np.ndarray) or isinstance(m_test,
                            pd.DataFrame) or isinstance(m_test, pd.Series):
                            shape_size = m_test.shape
                        elif isinstance(m_test, list):
                            shape_size = len(m_test)
                        else:
                            shape_size = 'any'
                        check_type = type(m_test)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('m_test')
                            writer = csv.writer(f)
                            writer.writerow(['m_test', 432, check_type,
                                shape_size])
                    if 'sum' not in TANGSHAN:
                        import csv
                        if isinstance(sum, np.ndarray) or isinstance(sum,
                            pd.DataFrame) or isinstance(sum, pd.Series):
                            shape_size = sum.shape
                        elif isinstance(sum, list):
                            shape_size = len(sum)
                        else:
                            shape_size = 'any'
                        check_type = type(sum)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('sum')
                            writer = csv.writer(f)
                            writer.writerow(['sum', 432, check_type,
                                shape_size])
                    y_size = y_test.shape[1]
                    if 'y_size' not in TANGSHAN:
                        import csv
                        if isinstance(y_size, np.ndarray) or isinstance(y_size,
                            pd.DataFrame) or isinstance(y_size, pd.Series):
                            shape_size = y_size.shape
                        elif isinstance(y_size, list):
                            shape_size = len(y_size)
                        else:
                            shape_size = 'any'
                        check_type = type(y_size)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('y_size')
                            writer = csv.writer(f)
                            writer.writerow(['y_size', 433, check_type,
                                shape_size])
                    pct = matched / m_test
                    if 'pct' not in TANGSHAN:
                        import csv
                        if isinstance(pct, np.ndarray) or isinstance(pct,
                            pd.DataFrame) or isinstance(pct, pd.Series):
                            shape_size = pct.shape
                        elif isinstance(pct, list):
                            shape_size = len(pct)
                        else:
                            shape_size = 'any'
                        check_type = type(pct)
                        with open('tas.csv', 'a+') as f:
                            TANGSHAN.append('pct')
                            writer = csv.writer(f)
                            writer.writerow(['pct', 434, check_type,
                                shape_size])
    if 'i' not in TANGSHAN:
        import csv
        if isinstance(i, np.ndarray) or isinstance(i, pd.DataFrame
            ) or isinstance(i, pd.Series):
            shape_size = i.shape
        elif isinstance(i, list):
            shape_size = len(i)
        else:
            shape_size = 'any'
        check_type = type(i)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('i')
            writer = csv.writer(f)
            writer.writerow(['i', 391, check_type, shape_size])
    etscost.append(tcost[-1])
    etrcost.append(cost[-1])
    if 'etrcost' not in TANGSHAN:
        import csv
        if isinstance(etrcost, np.ndarray) or isinstance(etrcost, pd.DataFrame
            ) or isinstance(etrcost, pd.Series):
            shape_size = etrcost.shape
        elif isinstance(etrcost, list):
            shape_size = len(etrcost)
        else:
            shape_size = 'any'
        check_type = type(etrcost)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('etrcost')
            writer = csv.writer(f)
            writer.writerow(['etrcost', 436, check_type, shape_size])
print('Preparing Data for submission')
X_test = get_features(test_raw)
if 'test_raw' not in TANGSHAN:
    import csv
    if isinstance(test_raw, np.ndarray) or isinstance(test_raw, pd.DataFrame
        ) or isinstance(test_raw, pd.Series):
        shape_size = test_raw.shape
    elif isinstance(test_raw, list):
        shape_size = len(test_raw)
    else:
        shape_size = 'any'
    check_type = type(test_raw)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test_raw')
        writer = csv.writer(f)
        writer.writerow(['test_raw', 441, check_type, shape_size])
print('Running Test Data On Model')
A2, vectors, _ = forward_dnn_propagation(X_test.T, params)
if 'vectors' not in TANGSHAN:
    import csv
    if isinstance(vectors, np.ndarray) or isinstance(vectors, pd.DataFrame
        ) or isinstance(vectors, pd.Series):
        shape_size = vectors.shape
    elif isinstance(vectors, list):
        shape_size = len(vectors)
    else:
        shape_size = 'any'
    check_type = type(vectors)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('vectors')
        writer = csv.writer(f)
        writer.writerow(['vectors', 443, check_type, shape_size])
print('Output Vector Shape {}'.format(A2.shape))
data = np.clip(A2.T, 0, 1)
if 'A2' not in TANGSHAN:
    import csv
    if isinstance(A2, np.ndarray) or isinstance(A2, pd.DataFrame
        ) or isinstance(A2, pd.Series):
        shape_size = A2.shape
    elif isinstance(A2, list):
        shape_size = len(A2)
    else:
        shape_size = 'any'
    check_type = type(A2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('A2')
        writer = csv.writer(f)
        writer.writerow(['A2', 446, check_type, shape_size])
data = data.argmax(axis=1)
print(data.shape)
data = np.reshape(data, (-1, 1))
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
        writer.writerow(['data', 451, check_type, shape_size])
print('Prepared Output Vector Shape {}'.format(data.shape))
index = np.reshape(np.arange(1, data.shape[0] + 1), (-1, 1))
if 'index' not in TANGSHAN:
    import csv
    if isinstance(index, np.ndarray) or isinstance(index, pd.DataFrame
        ) or isinstance(index, pd.Series):
        shape_size = index.shape
    elif isinstance(index, list):
        shape_size = len(index)
    else:
        shape_size = 'any'
    check_type = type(index)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('index')
        writer = csv.writer(f)
        writer.writerow(['index', 454, check_type, shape_size])
s1 = pd.Series(data[:, (0)], index=index[:, (0)])
if 's1' not in TANGSHAN:
    import csv
    if isinstance(s1, np.ndarray) or isinstance(s1, pd.DataFrame
        ) or isinstance(s1, pd.Series):
        shape_size = s1.shape
    elif isinstance(s1, list):
        shape_size = len(s1)
    else:
        shape_size = 'any'
    check_type = type(s1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('s1')
        writer = csv.writer(f)
        writer.writerow(['s1', 455, check_type, shape_size])
s0 = pd.Series(index[:, (0)])
if 's0' not in TANGSHAN:
    import csv
    if isinstance(s0, np.ndarray) or isinstance(s0, pd.DataFrame
        ) or isinstance(s0, pd.Series):
        shape_size = s0.shape
    elif isinstance(s0, list):
        shape_size = len(s0)
    else:
        shape_size = 'any'
    check_type = type(s0)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('s0')
        writer = csv.writer(f)
        writer.writerow(['s0', 456, check_type, shape_size])
df = pd.DataFrame(data=s1, index=index[:, (0)])
df.index.name = 'ImageId'
df.columns = ['Label']
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
        writer.writerow(['df', 459, check_type, shape_size])
df.replace([np.inf, -np.inf, np.nan], 0)
df = df.astype(int)
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
        writer.writerow(['int', 461, check_type, shape_size])
file_name = 'deep_nn.csv'
print('Saving Data to [{}]'.format(file_name))
df.to_csv(file_name, sep=',')
if 'file_name' not in TANGSHAN:
    import csv
    if isinstance(file_name, np.ndarray) or isinstance(file_name, pd.DataFrame
        ) or isinstance(file_name, pd.Series):
        shape_size = file_name.shape
    elif isinstance(file_name, list):
        shape_size = len(file_name)
    else:
        shape_size = 'any'
    check_type = type(file_name)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('file_name')
        writer = csv.writer(f)
        writer.writerow(['file_name', 464, check_type, shape_size])
print('========= End ===========')
