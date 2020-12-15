import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs
import matplotlib
import matplotlib.pyplot as plot
import pylab
import datetime
import re
df = pd.read_csv('../input/train.csv', usecols=['is_booking',
    'srch_adults_cnt', 'srch_destination_id', 'srch_ci', 'srch_co',
    'hotel_cluster'], chunksize=1000)
df = pd.concat(df, ignore_index=True)
df = df.groupby(['is_booking']).get_group(1)
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
        writer.writerow(['df', 23, check_type, shape_size])
dfx = df.ix[:, ('hotel_cluster')]
ylabel = dfx.value_counts()
ylabel = ylabel.index
if 'ylabel' not in TANGSHAN:
    import csv
    if isinstance(ylabel, np.ndarray) or isinstance(ylabel, pd.DataFrame
        ) or isinstance(ylabel, pd.Series):
        shape_size = ylabel.shape
    elif isinstance(ylabel, list):
        shape_size = len(ylabel)
    else:
        shape_size = 'any'
    check_type = type(ylabel)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ylabel')
        writer = csv.writer(f)
        writer.writerow(['ylabel', 27, check_type, shape_size])
y = dfx.as_matrix()
if 'y' not in TANGSHAN:
    import csv
    if isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame) or isinstance(y
        , pd.Series):
        shape_size = y.shape
    elif isinstance(y, list):
        shape_size = len(y)
    else:
        shape_size = 'any'
    check_type = type(y)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y')
        writer = csv.writer(f)
        writer.writerow(['y', 28, check_type, shape_size])
dfx = df.ix[:, ('srch_adults_cnt')]
x1 = dfx.as_matrix()
mu = np.mean(x1)
if 'x1' not in TANGSHAN:
    import csv
    if isinstance(x1, np.ndarray) or isinstance(x1, pd.DataFrame
        ) or isinstance(x1, pd.Series):
        shape_size = x1.shape
    elif isinstance(x1, list):
        shape_size = len(x1)
    else:
        shape_size = 'any'
    check_type = type(x1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x1')
        writer = csv.writer(f)
        writer.writerow(['x1', 33, check_type, shape_size])
s = np.amax(x1) - np.amin(x1)
x1 = (x1 - mu) / s
dfx = df.ix[:, ('srch_destination_id')]
x2 = dfx.as_matrix()
if 'x2' not in TANGSHAN:
    import csv
    if isinstance(x2, np.ndarray) or isinstance(x2, pd.DataFrame
        ) or isinstance(x2, pd.Series):
        shape_size = x2.shape
    elif isinstance(x2, list):
        shape_size = len(x2)
    else:
        shape_size = 'any'
    check_type = type(x2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x2')
        writer = csv.writer(f)
        writer.writerow(['x2', 38, check_type, shape_size])
mu = np.mean(x2)
if 'mu' not in TANGSHAN:
    import csv
    if isinstance(mu, np.ndarray) or isinstance(mu, pd.DataFrame
        ) or isinstance(mu, pd.Series):
        shape_size = mu.shape
    elif isinstance(mu, list):
        shape_size = len(mu)
    else:
        shape_size = 'any'
    check_type = type(mu)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('mu')
        writer = csv.writer(f)
        writer.writerow(['mu', 39, check_type, shape_size])
s = np.amax(x2) - np.amin(x2)
x2 = (x2 - mu) / s
dfx = df.ix[:, ('srch_ci')]
dfx = pd.to_datetime(dfx)
if 'dfx' not in TANGSHAN:
    import csv
    if isinstance(dfx, np.ndarray) or isinstance(dfx, pd.DataFrame
        ) or isinstance(dfx, pd.Series):
        shape_size = dfx.shape
    elif isinstance(dfx, list):
        shape_size = len(dfx)
    else:
        shape_size = 'any'
    check_type = type(dfx)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dfx')
        writer = csv.writer(f)
        writer.writerow(['dfx', 45, check_type, shape_size])
ci = dfx.dt.year * 365 + dfx.dt.month * 30 + dfx.dt.day
dfx = df.ix[:, ('srch_co')]
dfx = pd.to_datetime(dfx)
co = dfx.dt.year * 365 + dfx.dt.month * 30 + dfx.dt.day
x3 = co - ci
if 'co' not in TANGSHAN:
    import csv
    if isinstance(co, np.ndarray) or isinstance(co, pd.DataFrame
        ) or isinstance(co, pd.Series):
        shape_size = co.shape
    elif isinstance(co, list):
        shape_size = len(co)
    else:
        shape_size = 'any'
    check_type = type(co)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('co')
        writer = csv.writer(f)
        writer.writerow(['co', 51, check_type, shape_size])
if 'ci' not in TANGSHAN:
    import csv
    if isinstance(ci, np.ndarray) or isinstance(ci, pd.DataFrame
        ) or isinstance(ci, pd.Series):
        shape_size = ci.shape
    elif isinstance(ci, list):
        shape_size = len(ci)
    else:
        shape_size = 'any'
    check_type = type(ci)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ci')
        writer = csv.writer(f)
        writer.writerow(['ci', 51, check_type, shape_size])
x3 = x3.as_matrix()
mu = np.mean(x3)
s = np.amax(x3) - np.amin(x3)
if 'x3' not in TANGSHAN:
    import csv
    if isinstance(x3, np.ndarray) or isinstance(x3, pd.DataFrame
        ) or isinstance(x3, pd.Series):
        shape_size = x3.shape
    elif isinstance(x3, list):
        shape_size = len(x3)
    else:
        shape_size = 'any'
    check_type = type(x3)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x3')
        writer = csv.writer(f)
        writer.writerow(['x3', 55, check_type, shape_size])
x3 = (x3 - mu) / s
if 's' not in TANGSHAN:
    import csv
    if isinstance(s, np.ndarray) or isinstance(s, pd.DataFrame) or isinstance(s
        , pd.Series):
        shape_size = s.shape
    elif isinstance(s, list):
        shape_size = len(s)
    else:
        shape_size = 'any'
    check_type = type(s)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('s')
        writer = csv.writer(f)
        writer.writerow(['s', 56, check_type, shape_size])
x0 = np.ones(len(y))
if 'x0' not in TANGSHAN:
    import csv
    if isinstance(x0, np.ndarray) or isinstance(x0, pd.DataFrame
        ) or isinstance(x0, pd.Series):
        shape_size = x0.shape
    elif isinstance(x0, list):
        shape_size = len(x0)
    else:
        shape_size = 'any'
    check_type = type(x0)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x0')
        writer = csv.writer(f)
        writer.writerow(['x0', 58, check_type, shape_size])
X = np.vstack((x0, x1, x2, x3))
print(X.shape)
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
        writer.writerow(['print', 62, check_type, shape_size])
if 'X' not in TANGSHAN:
    import csv
    if isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame) or isinstance(X
        , pd.Series):
        shape_size = X.shape
    elif isinstance(X, list):
        shape_size = len(X)
    else:
        shape_size = 'any'
    check_type = type(X)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('X')
        writer = csv.writer(f)
        writer.writerow(['X', 62, check_type, shape_size])
print('Saving X and y values...')
np.savetxt('Xvalue.out', X)
np.savetxt('yvalue.out', y)
print('Done!')
