import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse.csgraph import minimum_spanning_tree
if 'list' not in TANGSHAN:
    import csv
    if isinstance(list, np.ndarray) or isinstance(list, pd.DataFrame
        ) or isinstance(list, pd.Series):
        shape_size = list.shape
    elif isinstance(list, list):
        shape_size = len(list)
    else:
        shape_size = 'any'
    check_type = type(list)
    if 'gpu' not in TANGSHAN:
        import csv
        if isinstance(gpu, np.ndarray) or isinstance(gpu, pd.DataFrame
            ) or isinstance(gpu, pd.Series):
            shape_size = gpu.shape
        elif isinstance(gpu, list):
            shape_size = len(gpu)
        else:
            shape_size = 'any'
        check_type = type(gpu)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('gpu')
            writer = csv.writer(f)
            writer.writerow(['gpu', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('list')
        writer = csv.writer(f)
        writer.writerow(['list', 11, check_type, shape_size])
import pylab as pl
from matplotlib import collections as mc
import os
print(os.listdir('../input'))
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
        writer.writerow(['print', 16, check_type, shape_size])
start_time = time.time()
if 'start_time' not in TANGSHAN:
    import csv
    if isinstance(start_time, np.ndarray) or isinstance(start_time, pd.
        DataFrame) or isinstance(start_time, pd.Series):
        shape_size = start_time.shape
    elif isinstance(start_time, list):
        shape_size = len(start_time)
    else:
        shape_size = 'any'
    check_type = type(start_time)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('start_time')
        writer = csv.writer(f)
        writer.writerow(['start_time', 18, check_type, shape_size])
inp = '../input/train.csv'
if 'inp' not in TANGSHAN:
    import csv
    if isinstance(inp, np.ndarray) or isinstance(inp, pd.DataFrame
        ) or isinstance(inp, pd.Series):
        shape_size = inp.shape
    elif isinstance(inp, list):
        shape_size = len(inp)
    else:
        shape_size = 'any'
    check_type = type(inp)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('inp')
        writer = csv.writer(f)
        writer.writerow(['inp', 19, check_type, shape_size])
gpu = -1
if 'gpu' not in TANGSHAN:
    import csv
    if isinstance(gpu, np.ndarray) or isinstance(gpu, pd.DataFrame
        ) or isinstance(gpu, pd.Series):
        shape_size = gpu.shape
    elif isinstance(gpu, list):
        shape_size = len(gpu)
    else:
        shape_size = 'any'
    check_type = type(gpu)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('gpu')
        writer = csv.writer(f)
        writer.writerow(['gpu', 20, check_type, shape_size])


def rotate(px, py, angle, ox=0, oy=0):
    angle = np.radians(angle)
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def rotate_coordinates(df, angle=29):
    df['pickup_longitude'], df['pickup_latitude'] = rotate(df[
        'pickup_longitude'], df['pickup_latitude'], angle)
    df['dropoff_longitude'], df['dropoff_latitude'] = rotate(df[
        'dropoff_longitude'], df['dropoff_latitude'], angle)


def parse_hour(datetime):
    left, right = datetime.split(':', 1)
    left_left, h = left.split(' ', 1)
    return int(h)


def arg_closest(point, x, y):
    dist = np.abs(point[:, ([0])] - x) + np.abs(point[:, ([1])] - y)
    amin = dist.argmin(axis=1)
    return amin, dist[np.arange(len(point)), amin]


def save_graph(graph, data_points, name, title, random_colors=False):
    print('saving graph to %s' % name)
    x = data_points[:, (0)]
    y = data_points[:, (1)]
    lines = []
    dists = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j] != REALLY_BIG_NUM and (i < j or graph[i, j] ==
                REALLY_BIG_NUM):
                lines.append(((x[i], y[i]), (x[j], y[j])))
                dists.append(graph[i, j])
    if random_colors:
        colors = np.random.rand(len(lines), 3)
    else:
        dists = np.array(dists)
        min_dists = dists.min()
        colors = np.zeros((len(lines), 3))
        colors[:, (0)] = (dists - min_dists) / (dists.max() - min_dists)
    lc = mc.LineCollection(lines, colors=colors)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    fig.suptitle(title)
    fig.savefig(name, dpi=300)


coord_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude']
if 'coord_cols' not in TANGSHAN:
    import csv
    if isinstance(coord_cols, np.ndarray) or isinstance(coord_cols, pd.
        DataFrame) or isinstance(coord_cols, pd.Series):
        shape_size = coord_cols.shape
    elif isinstance(coord_cols, list):
        shape_size = len(coord_cols)
    else:
        shape_size = 'any'
    check_type = type(coord_cols)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('coord_cols')
        writer = csv.writer(f)
        writer.writerow(['coord_cols', 84, check_type, shape_size])
df = pd.read_csv(inp, nrows=150000, header=0, doublequote=False, quoting=3,
    usecols=coord_cols)
df = df.dropna()
df = df[df['pickup_longitude'].between(-74.4, -72.9) & df['pickup_latitude'
    ].between(40.5, 41.7) & df['dropoff_longitude'].between(-74.4, -72.9) &
    df['dropoff_latitude'].between(40.5, 41.7)]
coord1 = df[['pickup_longitude', 'pickup_latitude']].values
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
        writer.writerow(['df', 94, check_type, shape_size])
if 'coord1' not in TANGSHAN:
    import csv
    if isinstance(coord1, np.ndarray) or isinstance(coord1, pd.DataFrame
        ) or isinstance(coord1, pd.Series):
        shape_size = coord1.shape
    elif isinstance(coord1, list):
        shape_size = len(coord1)
    else:
        shape_size = 'any'
    check_type = type(coord1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('coord1')
        writer = csv.writer(f)
        writer.writerow(['coord1', 94, check_type, shape_size])
coord2 = df[['dropoff_longitude', 'dropoff_latitude']].values
if 'coord2' not in TANGSHAN:
    import csv
    if isinstance(coord2, np.ndarray) or isinstance(coord2, pd.DataFrame
        ) or isinstance(coord2, pd.Series):
        shape_size = coord2.shape
    elif isinstance(coord2, list):
        shape_size = len(coord2)
    else:
        shape_size = 'any'
    check_type = type(coord2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('coord2')
        writer = csv.writer(f)
        writer.writerow(['coord2', 95, check_type, shape_size])
all_coords = np.concatenate([coord1, coord2], axis=0)
all_coords = shuffle(all_coords)
if 'all_coords' not in TANGSHAN:
    import csv
    if isinstance(all_coords, np.ndarray) or isinstance(all_coords, pd.
        DataFrame) or isinstance(all_coords, pd.Series):
        shape_size = all_coords.shape
    elif isinstance(all_coords, list):
        shape_size = len(all_coords)
    else:
        shape_size = 'any'
    check_type = type(all_coords)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('all_coords')
        writer = csv.writer(f)
        writer.writerow(['all_coords', 97, check_type, shape_size])
x = all_coords[:, (0)]
y = all_coords[:, (1)]
x, y = rotate(x, y, angle=29)
min_x = x.min(axis=0)
if 'min_x' not in TANGSHAN:
    import csv
    if isinstance(min_x, np.ndarray) or isinstance(min_x, pd.DataFrame
        ) or isinstance(min_x, pd.Series):
        shape_size = min_x.shape
    elif isinstance(min_x, list):
        shape_size = len(min_x)
    else:
        shape_size = 'any'
    check_type = type(min_x)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('min_x')
        writer = csv.writer(f)
        writer.writerow(['min_x', 104, check_type, shape_size])
max_x = x.max(axis=0)
min_y = y.min(axis=0)
max_y = y.max(axis=0)
scale_x = max_x - min_x
if 'scale_x' not in TANGSHAN:
    import csv
    if isinstance(scale_x, np.ndarray) or isinstance(scale_x, pd.DataFrame
        ) or isinstance(scale_x, pd.Series):
        shape_size = scale_x.shape
    elif isinstance(scale_x, list):
        shape_size = len(scale_x)
    else:
        shape_size = 'any'
    check_type = type(scale_x)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('scale_x')
        writer = csv.writer(f)
        writer.writerow(['scale_x', 108, check_type, shape_size])
scale_y = max_y - min_y
if 'max_y' not in TANGSHAN:
    import csv
    if isinstance(max_y, np.ndarray) or isinstance(max_y, pd.DataFrame
        ) or isinstance(max_y, pd.Series):
        shape_size = max_y.shape
    elif isinstance(max_y, list):
        shape_size = len(max_y)
    else:
        shape_size = 'any'
    check_type = type(max_y)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('max_y')
        writer = csv.writer(f)
        writer.writerow(['max_y', 109, check_type, shape_size])
scale_diff = (scale_x - scale_y) / 2
if 'scale_diff' not in TANGSHAN:
    import csv
    if isinstance(scale_diff, np.ndarray) or isinstance(scale_diff, pd.
        DataFrame) or isinstance(scale_diff, pd.Series):
        shape_size = scale_diff.shape
    elif isinstance(scale_diff, list):
        shape_size = len(scale_diff)
    else:
        shape_size = 'any'
    check_type = type(scale_diff)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('scale_diff')
        writer = csv.writer(f)
        writer.writerow(['scale_diff', 110, check_type, shape_size])
if 'scale_y' not in TANGSHAN:
    import csv
    if isinstance(scale_y, np.ndarray) or isinstance(scale_y, pd.DataFrame
        ) or isinstance(scale_y, pd.Series):
        shape_size = scale_y.shape
    elif isinstance(scale_y, list):
        shape_size = len(scale_y)
    else:
        shape_size = 'any'
    check_type = type(scale_y)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('scale_y')
        writer = csv.writer(f)
        writer.writerow(['scale_y', 110, check_type, shape_size])
if scale_diff > 0:
    min_y -= scale_diff
    max_y += scale_diff
else:
    min_x += scale_diff
    max_x -= scale_diff
    if 'max_x' not in TANGSHAN:
        import csv
        if isinstance(max_x, np.ndarray) or isinstance(max_x, pd.DataFrame
            ) or isinstance(max_x, pd.Series):
            shape_size = max_x.shape
        elif isinstance(max_x, list):
            shape_size = len(max_x)
        else:
            shape_size = 'any'
        check_type = type(max_x)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('max_x')
            writer = csv.writer(f)
            writer.writerow(['max_x', 116, check_type, shape_size])
bins = []
x_edges = []
y_edges = []
for divide in (2, 8, 32, 256):
    hist = np.zeros((divide - 1, divide - 1))
    if 'divide' not in TANGSHAN:
        import csv
        if isinstance(divide, np.ndarray) or isinstance(divide, pd.DataFrame
            ) or isinstance(divide, pd.Series):
            shape_size = divide.shape
        elif isinstance(divide, list):
            shape_size = len(divide)
        else:
            shape_size = 'any'
        check_type = type(divide)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('divide')
            writer = csv.writer(f)
            writer.writerow(['divide', 122, check_type, shape_size])
    if 'hist' not in TANGSHAN:
        import csv
        if isinstance(hist, np.ndarray) or isinstance(hist, pd.DataFrame
            ) or isinstance(hist, pd.Series):
            shape_size = hist.shape
        elif isinstance(hist, list):
            shape_size = len(hist)
        else:
            shape_size = 'any'
        check_type = type(hist)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('hist')
            writer = csv.writer(f)
            writer.writerow(['hist', 122, check_type, shape_size])
    boundaries1 = np.linspace(min_x, max_x, num=divide)
    if 'boundaries1' not in TANGSHAN:
        import csv
        if isinstance(boundaries1, np.ndarray) or isinstance(boundaries1,
            pd.DataFrame) or isinstance(boundaries1, pd.Series):
            shape_size = boundaries1.shape
        elif isinstance(boundaries1, list):
            shape_size = len(boundaries1)
        else:
            shape_size = 'any'
        check_type = type(boundaries1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('boundaries1')
            writer = csv.writer(f)
            writer.writerow(['boundaries1', 123, check_type, shape_size])
    boundaries2 = np.linspace(min_y, max_y, num=divide)
    if 'min_y' not in TANGSHAN:
        import csv
        if isinstance(min_y, np.ndarray) or isinstance(min_y, pd.DataFrame
            ) or isinstance(min_y, pd.Series):
            shape_size = min_y.shape
        elif isinstance(min_y, list):
            shape_size = len(min_y)
        else:
            shape_size = 'any'
        check_type = type(min_y)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('min_y')
            writer = csv.writer(f)
            writer.writerow(['min_y', 124, check_type, shape_size])
    bins.append(hist)
    if 'bins' not in TANGSHAN:
        import csv
        if isinstance(bins, np.ndarray) or isinstance(bins, pd.DataFrame
            ) or isinstance(bins, pd.Series):
            shape_size = bins.shape
        elif isinstance(bins, list):
            shape_size = len(bins)
        else:
            shape_size = 'any'
        check_type = type(bins)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('bins')
            writer = csv.writer(f)
            writer.writerow(['bins', 125, check_type, shape_size])
    x_edges.append(boundaries1)
    y_edges.append(boundaries2)
    if 'boundaries2' not in TANGSHAN:
        import csv
        if isinstance(boundaries2, np.ndarray) or isinstance(boundaries2,
            pd.DataFrame) or isinstance(boundaries2, pd.Series):
            shape_size = boundaries2.shape
        elif isinstance(boundaries2, list):
            shape_size = len(boundaries2)
        else:
            shape_size = 'any'
        check_type = type(boundaries2)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('boundaries2')
            writer = csv.writer(f)
            writer.writerow(['boundaries2', 127, check_type, shape_size])
    if 'y_edges' not in TANGSHAN:
        import csv
        if isinstance(y_edges, np.ndarray) or isinstance(y_edges, pd.DataFrame
            ) or isinstance(y_edges, pd.Series):
            shape_size = y_edges.shape
        elif isinstance(y_edges, list):
            shape_size = len(y_edges)
        else:
            shape_size = 'any'
        check_type = type(y_edges)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y_edges')
            writer = csv.writer(f)
            writer.writerow(['y_edges', 127, check_type, shape_size])
for b, x_e, y_e in zip(bins, x_edges, y_edges):
    c1 = np.searchsorted(x_e, x) - 1
    if 'x' not in TANGSHAN:
        import csv
        if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame
            ) or isinstance(x, pd.Series):
            shape_size = x.shape
        elif isinstance(x, list):
            shape_size = len(x)
        else:
            shape_size = 'any'
        check_type = type(x)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('x')
            writer = csv.writer(f)
            writer.writerow(['x', 131, check_type, shape_size])
    c2 = np.searchsorted(y_e, y) - 1
    for n, m in zip(c1, c2):
        b[n, m] += 1
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
                writer.writerow(['m', 134, check_type, shape_size])
    if 'c2' not in TANGSHAN:
        import csv
        if isinstance(c2, np.ndarray) or isinstance(c2, pd.DataFrame
            ) or isinstance(c2, pd.Series):
            shape_size = c2.shape
        elif isinstance(c2, list):
            shape_size = len(c2)
        else:
            shape_size = 'any'
        check_type = type(c2)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('c2')
            writer = csv.writer(f)
            writer.writerow(['c2', 133, check_type, shape_size])
    if 'c1' not in TANGSHAN:
        import csv
        if isinstance(c1, np.ndarray) or isinstance(c1, pd.DataFrame
            ) or isinstance(c1, pd.Series):
            shape_size = c1.shape
        elif isinstance(c1, list):
            shape_size = len(c1)
        else:
            shape_size = 'any'
        check_type = type(c1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('c1')
            writer = csv.writer(f)
            writer.writerow(['c1', 133, check_type, shape_size])
if 'b' not in TANGSHAN:
    import csv
    if isinstance(b, np.ndarray) or isinstance(b, pd.DataFrame) or isinstance(b
        , pd.Series):
        shape_size = b.shape
    elif isinstance(b, list):
        shape_size = len(b)
    else:
        shape_size = 'any'
    check_type = type(b)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('b')
        writer = csv.writer(f)
        writer.writerow(['b', 130, check_type, shape_size])
if 'y_e' not in TANGSHAN:
    import csv
    if isinstance(y_e, np.ndarray) or isinstance(y_e, pd.DataFrame
        ) or isinstance(y_e, pd.Series):
        shape_size = y_e.shape
    elif isinstance(y_e, list):
        shape_size = len(y_e)
    else:
        shape_size = 'any'
    check_type = type(y_e)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_e')
        writer = csv.writer(f)
        writer.writerow(['y_e', 130, check_type, shape_size])
if 'x_edges' not in TANGSHAN:
    import csv
    if isinstance(x_edges, np.ndarray) or isinstance(x_edges, pd.DataFrame
        ) or isinstance(x_edges, pd.Series):
        shape_size = x_edges.shape
    elif isinstance(x_edges, list):
        shape_size = len(x_edges)
    else:
        shape_size = 'any'
    check_type = type(x_edges)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x_edges')
        writer = csv.writer(f)
        writer.writerow(['x_edges', 130, check_type, shape_size])
print('counting done')
min_freq = 100
if 'min_freq' not in TANGSHAN:
    import csv
    if isinstance(min_freq, np.ndarray) or isinstance(min_freq, pd.DataFrame
        ) or isinstance(min_freq, pd.Series):
        shape_size = min_freq.shape
    elif isinstance(min_freq, list):
        shape_size = len(min_freq)
    else:
        shape_size = 'any'
    check_type = type(min_freq)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('min_freq')
        writer = csv.writer(f)
        writer.writerow(['min_freq', 139, check_type, shape_size])
points = []
for b, x_e, y_e in zip(bins, x_edges, y_edges):
    for i, x1 in enumerate(x_e[:-1]):
        x2 = x_e[i + 1]
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
                writer.writerow(['x2', 143, check_type, shape_size])
        x12 = (x1 + x2) / 2
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
                writer.writerow(['x1', 144, check_type, shape_size])
        for j, y1 in enumerate(y_e[:-1]):
            y2 = y_e[j + 1]
            y12 = (y1 + y2) / 2
            if 'y2' not in TANGSHAN:
                import csv
                if isinstance(y2, np.ndarray) or isinstance(y2, pd.DataFrame
                    ) or isinstance(y2, pd.Series):
                    shape_size = y2.shape
                elif isinstance(y2, list):
                    shape_size = len(y2)
                else:
                    shape_size = 'any'
                check_type = type(y2)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('y2')
                    writer = csv.writer(f)
                    writer.writerow(['y2', 147, check_type, shape_size])
            if b[i, j] > min_freq:
                points.append([x12, y12])
                if 'x12' not in TANGSHAN:
                    import csv
                    if isinstance(x12, np.ndarray) or isinstance(x12, pd.
                        DataFrame) or isinstance(x12, pd.Series):
                        shape_size = x12.shape
                    elif isinstance(x12, list):
                        shape_size = len(x12)
                    else:
                        shape_size = 'any'
                    check_type = type(x12)
                    with open('tas.csv', 'a+') as f:
                        TANGSHAN.append('x12')
                        writer = csv.writer(f)
                        writer.writerow(['x12', 149, check_type, shape_size])
                if 'y12' not in TANGSHAN:
                    import csv
                    if isinstance(y12, np.ndarray) or isinstance(y12, pd.
                        DataFrame) or isinstance(y12, pd.Series):
                        shape_size = y12.shape
                    elif isinstance(y12, list):
                        shape_size = len(y12)
                    else:
                        shape_size = 'any'
                    check_type = type(y12)
                    with open('tas.csv', 'a+') as f:
                        TANGSHAN.append('y12')
                        writer = csv.writer(f)
                        writer.writerow(['y12', 149, check_type, shape_size])
        if 'y1' not in TANGSHAN:
            import csv
            if isinstance(y1, np.ndarray) or isinstance(y1, pd.DataFrame
                ) or isinstance(y1, pd.Series):
                shape_size = y1.shape
            elif isinstance(y1, list):
                shape_size = len(y1)
            else:
                shape_size = 'any'
            check_type = type(y1)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('y1')
                writer = csv.writer(f)
                writer.writerow(['y1', 145, check_type, shape_size])
        if 'enumerate' not in TANGSHAN:
            import csv
            if isinstance(enumerate, np.ndarray) or isinstance(enumerate,
                pd.DataFrame) or isinstance(enumerate, pd.Series):
                shape_size = enumerate.shape
            elif isinstance(enumerate, list):
                shape_size = len(enumerate)
            else:
                shape_size = 'any'
            check_type = type(enumerate)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('enumerate')
                writer = csv.writer(f)
                writer.writerow(['enumerate', 145, check_type, shape_size])
    if 'x_e' not in TANGSHAN:
        import csv
        if isinstance(x_e, np.ndarray) or isinstance(x_e, pd.DataFrame
            ) or isinstance(x_e, pd.Series):
            shape_size = x_e.shape
        elif isinstance(x_e, list):
            shape_size = len(x_e)
        else:
            shape_size = 'any'
        check_type = type(x_e)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('x_e')
            writer = csv.writer(f)
            writer.writerow(['x_e', 142, check_type, shape_size])
print('number of points', len(points))
data_points = np.array(points)
if 'points' not in TANGSHAN:
    import csv
    if isinstance(points, np.ndarray) or isinstance(points, pd.DataFrame
        ) or isinstance(points, pd.Series):
        shape_size = points.shape
    elif isinstance(points, list):
        shape_size = len(points)
    else:
        shape_size = 'any'
    check_type = type(points)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('points')
        writer = csv.writer(f)
        writer.writerow(['points', 152, check_type, shape_size])
REALLY_BIG_NUM = 10000000


def add_symmetric_arcs(graph):
    for i in range(len(graph)):
        for j in range(len(graph)):
            graph[i, j] = min(graph[i, j], graph[j, i])
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
                    writer.writerow(['i', 163, check_type, shape_size])


def change_edge_values(graph, from_value, to_value):
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j] == from_value:
                graph[i, j] = to_value
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
                writer.writerow(['range', 167, check_type, shape_size])


def compute_mst(graph):
    dim = len(graph)
    add_symmetric_arcs(graph)
    change_edge_values(graph, REALLY_BIG_NUM, 0)
    spanning = minimum_spanning_tree(graph, overwrite=True).todense()
    change_edge_values(spanning, 0, REALLY_BIG_NUM)
    add_symmetric_arcs(spanning)
    return spanning


dim = len(data_points)
ows = np.arange(dim)
if 'ows' not in TANGSHAN:
    import csv
    if isinstance(ows, np.ndarray) or isinstance(ows, pd.DataFrame
        ) or isinstance(ows, pd.Series):
        shape_size = ows.shape
    elif isinstance(ows, list):
        shape_size = len(ows)
    else:
        shape_size = 'any'
    check_type = type(ows)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ows')
        writer = csv.writer(f)
        writer.writerow(['ows', 186, check_type, shape_size])
x = data_points[:, (0)]
y = data_points[:, (1)]
x_diff = x.reshape(dim, 1) - x.reshape(1, dim)
y_diff = y.reshape(dim, 1) - y.reshape(1, dim)
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
        writer.writerow(['y', 193, check_type, shape_size])
coord_scaling = (np.std(x_diff) + np.std(y_diff)) / 2
x_diff /= coord_scaling
y_diff /= coord_scaling
if 'coord_scaling' not in TANGSHAN:
    import csv
    if isinstance(coord_scaling, np.ndarray) or isinstance(coord_scaling,
        pd.DataFrame) or isinstance(coord_scaling, pd.Series):
        shape_size = coord_scaling.shape
    elif isinstance(coord_scaling, list):
        shape_size = len(coord_scaling)
    else:
        shape_size = 'any'
    check_type = type(coord_scaling)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('coord_scaling')
        writer = csv.writer(f)
        writer.writerow(['coord_scaling', 198, check_type, shape_size])
data_points /= coord_scaling
if 'data_points' not in TANGSHAN:
    import csv
    if isinstance(data_points, np.ndarray) or isinstance(data_points, pd.
        DataFrame) or isinstance(data_points, pd.Series):
        shape_size = data_points.shape
    elif isinstance(data_points, list):
        shape_size = len(data_points)
    else:
        shape_size = 'any'
    check_type = type(data_points)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('data_points')
        writer = csv.writer(f)
        writer.writerow(['data_points', 199, check_type, shape_size])
abs_diff = np.abs(x_diff) + np.abs(y_diff)
dist = np.sqrt(0.0 + np.square(x_diff) + np.square(y_diff))
if 'y_diff' not in TANGSHAN:
    import csv
    if isinstance(y_diff, np.ndarray) or isinstance(y_diff, pd.DataFrame
        ) or isinstance(y_diff, pd.Series):
        shape_size = y_diff.shape
    elif isinstance(y_diff, list):
        shape_size = len(y_diff)
    else:
        shape_size = 'any'
    check_type = type(y_diff)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y_diff')
        writer = csv.writer(f)
        writer.writerow(['y_diff', 202, check_type, shape_size])
abs_diff += 0.0001 * abs_diff * np.random.normal(size=len(dist))
if 'dist' not in TANGSHAN:
    import csv
    if isinstance(dist, np.ndarray) or isinstance(dist, pd.DataFrame
        ) or isinstance(dist, pd.Series):
        shape_size = dist.shape
    elif isinstance(dist, list):
        shape_size = len(dist)
    else:
        shape_size = 'any'
    check_type = type(dist)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dist')
        writer = csv.writer(f)
        writer.writerow(['dist', 205, check_type, shape_size])
angles = np.arctan2(y_diff, x_diff)
if 'x_diff' not in TANGSHAN:
    import csv
    if isinstance(x_diff, np.ndarray) or isinstance(x_diff, pd.DataFrame
        ) or isinstance(x_diff, pd.Series):
        shape_size = x_diff.shape
    elif isinstance(x_diff, list):
        shape_size = len(x_diff)
    else:
        shape_size = 'any'
    check_type = type(x_diff)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x_diff')
        writer = csv.writer(f)
        writer.writerow(['x_diff', 208, check_type, shape_size])
graph = np.empty((dim, dim))
graph.fill(REALLY_BIG_NUM)
if 'REALLY_BIG_NUM' not in TANGSHAN:
    import csv
    if isinstance(REALLY_BIG_NUM, np.ndarray) or isinstance(REALLY_BIG_NUM,
        pd.DataFrame) or isinstance(REALLY_BIG_NUM, pd.Series):
        shape_size = REALLY_BIG_NUM.shape
    elif isinstance(REALLY_BIG_NUM, list):
        shape_size = len(REALLY_BIG_NUM)
    else:
        shape_size = 'any'
    check_type = type(REALLY_BIG_NUM)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('REALLY_BIG_NUM')
        writer = csv.writer(f)
        writer.writerow(['REALLY_BIG_NUM', 212, check_type, shape_size])
for case, a in ((np.pi / 2, angles), (-np.pi / 2, angles)):
    criterion = (0.01 + np.abs(a - case)) * dist
    closest = np.argsort(criterion, axis=1)[:, 1:2]
    if 'criterion' not in TANGSHAN:
        import csv
        if isinstance(criterion, np.ndarray) or isinstance(criterion, pd.
            DataFrame) or isinstance(criterion, pd.Series):
            shape_size = criterion.shape
        elif isinstance(criterion, list):
            shape_size = len(criterion)
        else:
            shape_size = 'any'
        check_type = type(criterion)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('criterion')
            writer = csv.writer(f)
            writer.writerow(['criterion', 216, check_type, shape_size])
    for i, c in enumerate(closest):
        graph[i, c] = abs_diff[i, c]
        if 'abs_diff' not in TANGSHAN:
            import csv
            if isinstance(abs_diff, np.ndarray) or isinstance(abs_diff, pd.
                DataFrame) or isinstance(abs_diff, pd.Series):
                shape_size = abs_diff.shape
            elif isinstance(abs_diff, list):
                shape_size = len(abs_diff)
            else:
                shape_size = 'any'
            check_type = type(abs_diff)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('abs_diff')
                writer = csv.writer(f)
                writer.writerow(['abs_diff', 218, check_type, shape_size])
        if 'c' not in TANGSHAN:
            import csv
            if isinstance(c, np.ndarray) or isinstance(c, pd.DataFrame
                ) or isinstance(c, pd.Series):
                shape_size = c.shape
            elif isinstance(c, list):
                shape_size = len(c)
            else:
                shape_size = 'any'
            check_type = type(c)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('c')
                writer = csv.writer(f)
                writer.writerow(['c', 218, check_type, shape_size])
mst1 = compute_mst(graph)
graph = np.empty((dim, dim))
graph.fill(REALLY_BIG_NUM)
if 'graph' not in TANGSHAN:
    import csv
    if isinstance(graph, np.ndarray) or isinstance(graph, pd.DataFrame
        ) or isinstance(graph, pd.Series):
        shape_size = graph.shape
    elif isinstance(graph, list):
        shape_size = len(graph)
    else:
        shape_size = 'any'
    check_type = type(graph)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('graph')
        writer = csv.writer(f)
        writer.writerow(['graph', 224, check_type, shape_size])
for case, a in ((0, angles), (np.pi, np.abs(angles))):
    criterion = (0.01 + np.abs(a - case)) * dist
    closest = np.argsort(criterion, axis=1)[:, 1:2]
    for i, c in enumerate(closest):
        graph[i, c] = abs_diff[i, c]
    if 'closest' not in TANGSHAN:
        import csv
        if isinstance(closest, np.ndarray) or isinstance(closest, pd.DataFrame
            ) or isinstance(closest, pd.Series):
            shape_size = closest.shape
        elif isinstance(closest, list):
            shape_size = len(closest)
        else:
            shape_size = 'any'
        check_type = type(closest)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('closest')
            writer = csv.writer(f)
            writer.writerow(['closest', 228, check_type, shape_size])
if 'angles' not in TANGSHAN:
    import csv
    if isinstance(angles, np.ndarray) or isinstance(angles, pd.DataFrame
        ) or isinstance(angles, pd.Series):
        shape_size = angles.shape
    elif isinstance(angles, list):
        shape_size = len(angles)
    else:
        shape_size = 'any'
    check_type = type(angles)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('angles')
        writer = csv.writer(f)
        writer.writerow(['angles', 225, check_type, shape_size])
if 'a' not in TANGSHAN:
    import csv
    if isinstance(a, np.ndarray) or isinstance(a, pd.DataFrame) or isinstance(a
        , pd.Series):
        shape_size = a.shape
    elif isinstance(a, list):
        shape_size = len(a)
    else:
        shape_size = 'any'
    check_type = type(a)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('a')
        writer = csv.writer(f)
        writer.writerow(['a', 225, check_type, shape_size])
if 'case' not in TANGSHAN:
    import csv
    if isinstance(case, np.ndarray) or isinstance(case, pd.DataFrame
        ) or isinstance(case, pd.Series):
        shape_size = case.shape
    elif isinstance(case, list):
        shape_size = len(case)
    else:
        shape_size = 'any'
    check_type = type(case)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('case')
        writer = csv.writer(f)
        writer.writerow(['case', 225, check_type, shape_size])
mst2 = compute_mst(graph)
G = np.empty((dim, dim))
for i in range(dim):
    for j in range(dim):
        G[i, j] = min(mst1[i, j], mst2[i, j])
        if 'mst1' not in TANGSHAN:
            import csv
            if isinstance(mst1, np.ndarray) or isinstance(mst1, pd.DataFrame
                ) or isinstance(mst1, pd.Series):
                shape_size = mst1.shape
            elif isinstance(mst1, list):
                shape_size = len(mst1)
            else:
                shape_size = 'any'
            check_type = type(mst1)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('mst1')
                writer = csv.writer(f)
                writer.writerow(['mst1', 237, check_type, shape_size])
        if 'mst2' not in TANGSHAN:
            import csv
            if isinstance(mst2, np.ndarray) or isinstance(mst2, pd.DataFrame
                ) or isinstance(mst2, pd.Series):
                shape_size = mst2.shape
            elif isinstance(mst2, list):
                shape_size = len(mst2)
            else:
                shape_size = 'any'
            check_type = type(mst2)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('mst2')
                writer = csv.writer(f)
                writer.writerow(['mst2', 237, check_type, shape_size])
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
                writer.writerow(['min', 237, check_type, shape_size])
add_symmetric_arcs(G)
save_graph(G, data_points, 'nyc1.png', 'graph edges', random_colors=True)


def torch_dijsktra(gr):
    graph_dim = len(gr)
    graph_copy = [([None] * graph_dim) for _ in range(graph_dim)]
    distances = np.empty(graph_dim, dtype=np.float32)
    for i, edges in enumerate(gr):
        distances.fill(REALLY_BIG_NUM)
        distances[i] = 0
        torch_dist = graph_copy[i]
        torch_dist[i] = V(torch.zeros(1))
        for _ in range(graph_dim):
            v = distances.argmin()
            v_dist = torch_dist[v]
            distances[v] = np.inf
            for neighbor, d, min_d in gr[v]:
                new_d = v_dist + d.clamp(min=0.95, max=1.3) * min_d
                existing_d = torch_dist[neighbor]
                if existing_d is None or (new_d < existing_d).all():
                    torch_dist[neighbor] = new_d
                    distances[neighbor] = new_d.data.numpy()[0]
    return graph_copy


def loss(dist_matrix, closest_pickup, closest_dropoff, extra, scaling,
    y_true, clip=20):
    distances = V(torch.zeros(len(closest_pickup)))
    for n, (cp, cd) in enumerate(zip(closest_pickup, closest_dropoff)):
        distances[n] = dist_matrix[cp][cd]
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
                writer.writerow(['n', 290, check_type, shape_size])
    y_pred = 2.5 + (distances + extra) * scaling
    if 'scaling' not in TANGSHAN:
        import csv
        if isinstance(scaling, np.ndarray) or isinstance(scaling, pd.DataFrame
            ) or isinstance(scaling, pd.Series):
            shape_size = scaling.shape
        elif isinstance(scaling, list):
            shape_size = len(scaling)
        else:
            shape_size = 'any'
        check_type = type(scaling)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('scaling')
            writer = csv.writer(f)
            writer.writerow(['scaling', 292, check_type, shape_size])
    error = (y_pred - y_true).abs().clamp(max=clip).mean()
    if 'y_true' not in TANGSHAN:
        import csv
        if isinstance(y_true, np.ndarray) or isinstance(y_true, pd.DataFrame
            ) or isinstance(y_true, pd.Series):
            shape_size = y_true.shape
        elif isinstance(y_true, list):
            shape_size = len(y_true)
        else:
            shape_size = 'any'
        check_type = type(y_true)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y_true')
            writer = csv.writer(f)
            writer.writerow(['y_true', 293, check_type, shape_size])
    return error
    if 'error' not in TANGSHAN:
        import csv
        if isinstance(error, np.ndarray) or isinstance(error, pd.DataFrame
            ) or isinstance(error, pd.Series):
            shape_size = error.shape
        elif isinstance(error, list):
            shape_size = len(error)
        else:
            shape_size = 'any'
        check_type = type(error)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('error')
            writer = csv.writer(f)
            writer.writerow(['error', 295, check_type, shape_size])


fixed_fares = np.array([57.33, 49.8, 45.0, 52.0, 49.57, 56.8, 57.54, 49.15])
if 'fixed_fares' not in TANGSHAN:
    import csv
    if isinstance(fixed_fares, np.ndarray) or isinstance(fixed_fares, pd.
        DataFrame) or isinstance(fixed_fares, pd.Series):
        shape_size = fixed_fares.shape
    elif isinstance(fixed_fares, list):
        shape_size = len(fixed_fares)
    else:
        shape_size = 'any'
    check_type = type(fixed_fares)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('fixed_fares')
        writer = csv.writer(f)
        writer.writerow(['fixed_fares', 300, check_type, shape_size])
night = np.array([0, 1, 2, 3, 4, 5])
expected = 1000000
gen = pd.read_csv(inp, quoting=3, header=0, doublequote=False, chunksize=
    2048 * 32)
size = 0
batches = []
for df in gen:
    print('%i/%i' % (size, expected))
    df = df.dropna()
    hours = list(map(parse_hour, df['pickup_datetime']))
    if 'map' not in TANGSHAN:
        import csv
        if isinstance(map, np.ndarray) or isinstance(map, pd.DataFrame
            ) or isinstance(map, pd.Series):
            shape_size = map.shape
        elif isinstance(map, list):
            shape_size = len(map)
        else:
            shape_size = 'any'
        check_type = type(map)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('map')
            writer = csv.writer(f)
            writer.writerow(['map', 315, check_type, shape_size])
    fares = df['fare_amount'].values
    df = df[np.isin(hours, night, assume_unique=True) & np.isin(fares, 
    if 'fares' not in TANGSHAN:
        import csv
        if isinstance(fares, np.ndarray) or isinstance(fares, pd.DataFrame
            ) or isinstance(fares, pd.Series):
            shape_size = fares.shape
        elif isinstance(fares, list):
            shape_size = len(fares)
        else:
            shape_size = 'any'
        check_type = type(fares)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('fares')
            writer = csv.writer(f)
            writer.writerow(['fares', 318, check_type, shape_size]
                ), fixed_fares, assume_unique=True, invert=True) & (df[
                'passenger_count'] <= 3 & df['pickup_longitude'].between(-
                74.4, -72.9) & df['pickup_latitude'].between(40.5, 41.7) &
                df['dropoff_longitude'].between(-74.4, -72.9) & df[
                'dropoff_latitude'].between(40.5, 41.7) & df['fare_amount']
                .between(3, 250)).values]
    if 'night' not in TANGSHAN:
        import csv
        if isinstance(night, np.ndarray) or isinstance(night, pd.DataFrame
            ) or isinstance(night, pd.Series):
            shape_size = night.shape
        elif isinstance(night, list):
            shape_size = len(night)
        else:
            shape_size = 'any'
        check_type = type(night)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('night')
            writer = csv.writer(f)
            writer.writerow(['night', 317, check_type, shape_size])
    if 'hours' not in TANGSHAN:
        import csv
        if isinstance(hours, np.ndarray) or isinstance(hours, pd.DataFrame
            ) or isinstance(hours, pd.Series):
            shape_size = hours.shape
        elif isinstance(hours, list):
            shape_size = len(hours)
        else:
            shape_size = 'any'
        check_type = type(hours)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('hours')
            writer = csv.writer(f)
            writer.writerow(['hours', 317, check_type, shape_size])
    rotate_coordinates(df)
    pickup = df[['pickup_longitude', 'pickup_latitude']].values.astype(np.
        float32)
    if 'pickup' not in TANGSHAN:
        import csv
        if isinstance(pickup, np.ndarray) or isinstance(pickup, pd.DataFrame
            ) or isinstance(pickup, pd.Series):
            shape_size = pickup.shape
        elif isinstance(pickup, list):
            shape_size = len(pickup)
        else:
            shape_size = 'any'
        check_type = type(pickup)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pickup')
            writer = csv.writer(f)
            writer.writerow(['pickup', 328, check_type, shape_size])
    dropoff = df[['dropoff_longitude', 'dropoff_latitude']].values.astype(np
        .float32)
    targets = df['fare_amount'].values.astype(np.float32)
    batches.append((pickup, dropoff, targets))
    size += len(targets)
    if 'size' not in TANGSHAN:
        import csv
        if isinstance(size, np.ndarray) or isinstance(size, pd.DataFrame
            ) or isinstance(size, pd.Series):
            shape_size = size.shape
        elif isinstance(size, list):
            shape_size = len(size)
        else:
            shape_size = 'any'
        check_type = type(size)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('size')
            writer = csv.writer(f)
            writer.writerow(['size', 333, check_type, shape_size])
    if size >= expected:
        break
    if 'expected' not in TANGSHAN:
        import csv
        if isinstance(expected, np.ndarray) or isinstance(expected, pd.
            DataFrame) or isinstance(expected, pd.Series):
            shape_size = expected.shape
        elif isinstance(expected, list):
            shape_size = len(expected)
        else:
            shape_size = 'any'
        check_type = type(expected)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('expected')
            writer = csv.writer(f)
            writer.writerow(['expected', 334, check_type, shape_size])
if 'gen' not in TANGSHAN:
    import csv
    if isinstance(gen, np.ndarray) or isinstance(gen, pd.DataFrame
        ) or isinstance(gen, pd.Series):
        shape_size = gen.shape
    elif isinstance(gen, list):
        shape_size = len(gen)
    else:
        shape_size = 'any'
    check_type = type(gen)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('gen')
        writer = csv.writer(f)
        writer.writerow(['gen', 312, check_type, shape_size])
pickup, dropoff, targets = zip(*batches)
if 'zip' not in TANGSHAN:
    import csv
    if isinstance(zip, np.ndarray) or isinstance(zip, pd.DataFrame
        ) or isinstance(zip, pd.Series):
        shape_size = zip.shape
    elif isinstance(zip, list):
        shape_size = len(zip)
    else:
        shape_size = 'any'
    check_type = type(zip)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('zip')
        writer = csv.writer(f)
        writer.writerow(['zip', 337, check_type, shape_size])
if 'batches' not in TANGSHAN:
    import csv
    if isinstance(batches, np.ndarray) or isinstance(batches, pd.DataFrame
        ) or isinstance(batches, pd.Series):
        shape_size = batches.shape
    elif isinstance(batches, list):
        shape_size = len(batches)
    else:
        shape_size = 'any'
    check_type = type(batches)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('batches')
        writer = csv.writer(f)
        writer.writerow(['batches', 337, check_type, shape_size])
pickup = np.concatenate(pickup, axis=0)
dropoff = np.concatenate(dropoff, axis=0)
targets = np.concatenate(targets)
print('has read', size, 'rows')
pickup /= coord_scaling
dropoff /= coord_scaling
x = data_points[:, (0)].reshape((1, len(data_points)))
y = data_points[:, (1)].reshape((1, len(data_points)))
closest_pickup, d1 = arg_closest(pickup, x, y)
if 'd1' not in TANGSHAN:
    import csv
    if isinstance(d1, np.ndarray) or isinstance(d1, pd.DataFrame
        ) or isinstance(d1, pd.Series):
        shape_size = d1.shape
    elif isinstance(d1, list):
        shape_size = len(d1)
    else:
        shape_size = 'any'
    check_type = type(d1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('d1')
        writer = csv.writer(f)
        writer.writerow(['d1', 351, check_type, shape_size])
closest_dropoff, d2 = arg_closest(dropoff, x, y)
w = np.where(closest_pickup != closest_dropoff)[0]
closest_pickup = closest_pickup[w]
if 'closest_pickup' not in TANGSHAN:
    import csv
    if isinstance(closest_pickup, np.ndarray) or isinstance(closest_pickup,
        pd.DataFrame) or isinstance(closest_pickup, pd.Series):
        shape_size = closest_pickup.shape
    elif isinstance(closest_pickup, list):
        shape_size = len(closest_pickup)
    else:
        shape_size = 'any'
    check_type = type(closest_pickup)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('closest_pickup')
        writer = csv.writer(f)
        writer.writerow(['closest_pickup', 356, check_type, shape_size])
closest_dropoff = closest_dropoff[w]
if 'w' not in TANGSHAN:
    import csv
    if isinstance(w, np.ndarray) or isinstance(w, pd.DataFrame) or isinstance(w
        , pd.Series):
        shape_size = w.shape
    elif isinstance(w, list):
        shape_size = len(w)
    else:
        shape_size = 'any'
    check_type = type(w)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('w')
        writer = csv.writer(f)
        writer.writerow(['w', 357, check_type, shape_size])
d1 = d1[w]
d2 = d2[w]
if 'd2' not in TANGSHAN:
    import csv
    if isinstance(d2, np.ndarray) or isinstance(d2, pd.DataFrame
        ) or isinstance(d2, pd.Series):
        shape_size = d2.shape
    elif isinstance(d2, list):
        shape_size = len(d2)
    else:
        shape_size = 'any'
    check_type = type(d2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('d2')
        writer = csv.writer(f)
        writer.writerow(['d2', 359, check_type, shape_size])
targets = targets[w]
pickup = pickup[w]
dropoff = dropoff[w]
if 'dropoff' not in TANGSHAN:
    import csv
    if isinstance(dropoff, np.ndarray) or isinstance(dropoff, pd.DataFrame
        ) or isinstance(dropoff, pd.Series):
        shape_size = dropoff.shape
    elif isinstance(dropoff, list):
        shape_size = len(dropoff)
    else:
        shape_size = 'any'
    check_type = type(dropoff)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dropoff')
        writer = csv.writer(f)
        writer.writerow(['dropoff', 362, check_type, shape_size])
print(len(w), 'rows where grid(dropoff) != grid(pickup)')
ideal = np.abs(pickup - dropoff).sum(axis=1)
if 'ideal' not in TANGSHAN:
    import csv
    if isinstance(ideal, np.ndarray) or isinstance(ideal, pd.DataFrame
        ) or isinstance(ideal, pd.Series):
        shape_size = ideal.shape
    elif isinstance(ideal, list):
        shape_size = len(ideal)
    else:
        shape_size = 'any'
    check_type = type(ideal)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ideal')
        writer = csv.writer(f)
        writer.writerow(['ideal', 367, check_type, shape_size])
di = d1 + d2
hub_dist = np.abs(data_points[closest_pickup] - data_points[closest_dropoff]
    ).sum(axis=1)
ratio = (di + hub_dist) / ideal
if 'hub_dist' not in TANGSHAN:
    import csv
    if isinstance(hub_dist, np.ndarray) or isinstance(hub_dist, pd.DataFrame
        ) or isinstance(hub_dist, pd.Series):
        shape_size = hub_dist.shape
    elif isinstance(hub_dist, list):
        shape_size = len(hub_dist)
    else:
        shape_size = 'any'
    check_type = type(hub_dist)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('hub_dist')
        writer = csv.writer(f)
        writer.writerow(['hub_dist', 370, check_type, shape_size])
if 'ratio' not in TANGSHAN:
    import csv
    if isinstance(ratio, np.ndarray) or isinstance(ratio, pd.DataFrame
        ) or isinstance(ratio, pd.Series):
        shape_size = ratio.shape
    elif isinstance(ratio, list):
        shape_size = len(ratio)
    else:
        shape_size = 'any'
    check_type = type(ratio)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ratio')
        writer = csv.writer(f)
        writer.writerow(['ratio', 370, check_type, shape_size])
if 'di' not in TANGSHAN:
    import csv
    if isinstance(di, np.ndarray) or isinstance(di, pd.DataFrame
        ) or isinstance(di, pd.Series):
        shape_size = di.shape
    elif isinstance(di, list):
        shape_size = len(di)
    else:
        shape_size = 'any'
    check_type = type(di)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('di')
        writer = csv.writer(f)
        writer.writerow(['di', 370, check_type, shape_size])
print('ratios', ratio)
top = ratio.argsort()[:len(di) // 20]
if 'top' not in TANGSHAN:
    import csv
    if isinstance(top, np.ndarray) or isinstance(top, pd.DataFrame
        ) or isinstance(top, pd.Series):
        shape_size = top.shape
    elif isinstance(top, list):
        shape_size = len(top)
    else:
        shape_size = 'any'
    check_type = type(top)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('top')
        writer = csv.writer(f)
        writer.writerow(['top', 372, check_type, shape_size])
closest_pickup = closest_pickup[top]
closest_dropoff = closest_dropoff[top]
di = di[top].astype(np.float32)
targets = targets[top]
vcut = 16384
v_targets = torch.from_numpy(targets[:vcut])
if 'targets' not in TANGSHAN:
    import csv
    if isinstance(targets, np.ndarray) or isinstance(targets, pd.DataFrame
        ) or isinstance(targets, pd.Series):
        shape_size = targets.shape
    elif isinstance(targets, list):
        shape_size = len(targets)
    else:
        shape_size = 'any'
    check_type = type(targets)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('targets')
        writer = csv.writer(f)
        writer.writerow(['targets', 380, check_type, shape_size])
v_di = torch.from_numpy(di[:vcut])
if gpu >= 0:
    v_targets = v_targets.cuda(gpu)
    v_di = v_di.cuda(gpu)
    if 'v_di' not in TANGSHAN:
        import csv
        if isinstance(v_di, np.ndarray) or isinstance(v_di, pd.DataFrame
            ) or isinstance(v_di, pd.Series):
            shape_size = v_di.shape
        elif isinstance(v_di, list):
            shape_size = len(v_di)
        else:
            shape_size = 'any'
        check_type = type(v_di)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('v_di')
            writer = csv.writer(f)
            writer.writerow(['v_di', 384, check_type, shape_size])
v_targets = V(v_targets)
if 'v_targets' not in TANGSHAN:
    import csv
    if isinstance(v_targets, np.ndarray) or isinstance(v_targets, pd.DataFrame
        ) or isinstance(v_targets, pd.Series):
        shape_size = v_targets.shape
    elif isinstance(v_targets, list):
        shape_size = len(v_targets)
    else:
        shape_size = 'any'
    check_type = type(v_targets)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('v_targets')
        writer = csv.writer(f)
        writer.writerow(['v_targets', 385, check_type, shape_size])
v_di = V(v_di)
validation = closest_pickup[:vcut], closest_dropoff[:vcut], v_di, v_targets
targets = targets[vcut:]
closest_pickup = closest_pickup[vcut:]
closest_dropoff = closest_dropoff[vcut:]
if 'vcut' not in TANGSHAN:
    import csv
    if isinstance(vcut, np.ndarray) or isinstance(vcut, pd.DataFrame
        ) or isinstance(vcut, pd.Series):
        shape_size = vcut.shape
    elif isinstance(vcut, list):
        shape_size = len(vcut)
    else:
        shape_size = 'any'
    check_type = type(vcut)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('vcut')
        writer = csv.writer(f)
        writer.writerow(['vcut', 391, check_type, shape_size])
if 'closest_dropoff' not in TANGSHAN:
    import csv
    if isinstance(closest_dropoff, np.ndarray) or isinstance(closest_dropoff,
        pd.DataFrame) or isinstance(closest_dropoff, pd.Series):
        shape_size = closest_dropoff.shape
    elif isinstance(closest_dropoff, list):
        shape_size = len(closest_dropoff)
    else:
        shape_size = 'any'
    check_type = type(closest_dropoff)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('closest_dropoff')
        writer = csv.writer(f)
        writer.writerow(['closest_dropoff', 391, check_type, shape_size])
di = di[vcut:]
parameter_graph = []
parameters = []
if 'parameters' not in TANGSHAN:
    import csv
    if isinstance(parameters, np.ndarray) or isinstance(parameters, pd.
        DataFrame) or isinstance(parameters, pd.Series):
        shape_size = parameters.shape
    elif isinstance(parameters, list):
        shape_size = len(parameters)
    else:
        shape_size = 'any'
    check_type = type(parameters)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('parameters')
        writer = csv.writer(f)
        writer.writerow(['parameters', 396, check_type, shape_size])
for i in range(len(G)):
    edges = []
    parameter_graph.append(edges)
    for j in range(len(G)):
        if G[i][j] < REALLY_BIG_NUM:
            d = G[i][j]
            if 'd' not in TANGSHAN:
                import csv
                if isinstance(d, np.ndarray) or isinstance(d, pd.DataFrame
                    ) or isinstance(d, pd.Series):
                    shape_size = d.shape
                elif isinstance(d, list):
                    shape_size = len(d)
                else:
                    shape_size = 'any'
                check_type = type(d)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('d')
                    writer = csv.writer(f)
                    writer.writerow(['d', 402, check_type, shape_size])
            param = nn.Parameter(torch.Tensor([1]))
            if gpu >= 0:
                param = param.cuda(gpu)
            parameters.append(param)
            if 'param' not in TANGSHAN:
                import csv
                if isinstance(param, np.ndarray) or isinstance(param, pd.
                    DataFrame) or isinstance(param, pd.Series):
                    shape_size = param.shape
                elif isinstance(param, list):
                    shape_size = len(param)
                else:
                    shape_size = 'any'
                check_type = type(param)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('param')
                    writer = csv.writer(f)
                    writer.writerow(['param', 406, check_type, shape_size])
            edges.append((j, param, d))
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
                    writer.writerow(['j', 407, check_type, shape_size])
scaling = nn.Parameter(torch.Tensor([10]))
if gpu >= 0:
    scaling = scaling.cuda(gpu)
optimizer = torch.optim.Adam([scaling], lr=0.05)
if 'optimizer' not in TANGSHAN:
    import csv
    if isinstance(optimizer, np.ndarray) or isinstance(optimizer, pd.DataFrame
        ) or isinstance(optimizer, pd.Series):
        shape_size = optimizer.shape
    elif isinstance(optimizer, list):
        shape_size = len(optimizer)
    else:
        shape_size = 'any'
    check_type = type(optimizer)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('optimizer')
        writer = csv.writer(f)
        writer.writerow(['optimizer', 417, check_type, shape_size])
print('start optimizing with', len(targets), 'training rows')
max_duration = 3 * 60 * 60
batch_size = 256
switch = 1
if 'switch' not in TANGSHAN:
    import csv
    if isinstance(switch, np.ndarray) or isinstance(switch, pd.DataFrame
        ) or isinstance(switch, pd.Series):
        shape_size = switch.shape
    elif isinstance(switch, list):
        shape_size = len(switch)
    else:
        shape_size = 'any'
    check_type = type(switch)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('switch')
        writer = csv.writer(f)
        writer.writerow(['switch', 423, check_type, shape_size])
for epoch in range(4):
    if time.time() - start_time > max_duration:
        print('optimization timeout')
        break
    if epoch == switch:
        print('switching to optimization with all parameters')
        optimizer = torch.optim.SGD([{'params': [scaling]}, {'params':
            parameters}], lr=0.15)
    if 'epoch' not in TANGSHAN:
        import csv
        if isinstance(epoch, np.ndarray) or isinstance(epoch, pd.DataFrame
            ) or isinstance(epoch, pd.Series):
            shape_size = epoch.shape
        elif isinstance(epoch, list):
            shape_size = len(epoch)
        else:
            shape_size = 'any'
        check_type = type(epoch)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('epoch')
            writer = csv.writer(f)
            writer.writerow(['epoch', 430, check_type, shape_size])
    targets, closest_pickup, closest_dropoff, di = shuffle(targets,
        closest_pickup, closest_dropoff, di)
    for i in range(0, len(targets), batch_size):
        if time.time() - start_time > max_duration:
            break
        if 'max_duration' not in TANGSHAN:
            import csv
            if isinstance(max_duration, np.ndarray) or isinstance(max_duration,
                pd.DataFrame) or isinstance(max_duration, pd.Series):
                shape_size = max_duration.shape
            elif isinstance(max_duration, list):
                shape_size = len(max_duration)
            else:
                shape_size = 'any'
            check_type = type(max_duration)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('max_duration')
                writer = csv.writer(f)
                writer.writerow(['max_duration', 438, check_type, shape_size])
        closest_pickup_b = closest_pickup[i:i + batch_size]
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
                writer.writerow(['batch_size', 442, check_type, shape_size])
        closest_dropoff_b = closest_dropoff[i:i + batch_size]
        y_true = torch.from_numpy(targets[i:i + batch_size])
        di_b = torch.from_numpy(di[i:i + batch_size])
        if gpu >= 0:
            y_true = y_true.cuda(gpu)
            di_b = di_b.cuda(gpu)
            if 'di_b' not in TANGSHAN:
                import csv
                if isinstance(di_b, np.ndarray) or isinstance(di_b, pd.
                    DataFrame) or isinstance(di_b, pd.Series):
                    shape_size = di_b.shape
                elif isinstance(di_b, list):
                    shape_size = len(di_b)
                else:
                    shape_size = 'any'
                check_type = type(di_b)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('di_b')
                    writer = csv.writer(f)
                    writer.writerow(['di_b', 450, check_type, shape_size])
        y_true = V(y_true)
        di_b = V(di_b)
        dist_matrix = torch_dijsktra(parameter_graph)
        error = loss(dist_matrix, closest_pickup_b, closest_dropoff_b, di_b,
            scaling, y_true)
        if 'dist_matrix' not in TANGSHAN:
            import csv
            if isinstance(dist_matrix, np.ndarray) or isinstance(dist_matrix,
                pd.DataFrame) or isinstance(dist_matrix, pd.Series):
                shape_size = dist_matrix.shape
            elif isinstance(dist_matrix, list):
                shape_size = len(dist_matrix)
            else:
                shape_size = 'any'
            check_type = type(dist_matrix)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('dist_matrix')
                writer = csv.writer(f)
                writer.writerow(['dist_matrix', 456, check_type, shape_size])
        optimizer.zero_grad()
        error.backward()
        nn.utils.clip_grad_norm_(parameters, 10)
        optimizer.step()
        closest_pickup_b, closest_dropoff_b, di_b, y_true = validation
        if 'validation' not in TANGSHAN:
            import csv
            if isinstance(validation, np.ndarray) or isinstance(validation,
                pd.DataFrame) or isinstance(validation, pd.Series):
                shape_size = validation.shape
            elif isinstance(validation, list):
                shape_size = len(validation)
            else:
                shape_size = 'any'
            check_type = type(validation)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('validation')
                writer = csv.writer(f)
                writer.writerow(['validation', 465, check_type, shape_size])
        if 'closest_dropoff_b' not in TANGSHAN:
            import csv
            if isinstance(closest_dropoff_b, np.ndarray) or isinstance(
                closest_dropoff_b, pd.DataFrame) or isinstance(
                closest_dropoff_b, pd.Series):
                shape_size = closest_dropoff_b.shape
            elif isinstance(closest_dropoff_b, list):
                shape_size = len(closest_dropoff_b)
            else:
                shape_size = 'any'
            check_type = type(closest_dropoff_b)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('closest_dropoff_b')
                writer = csv.writer(f)
                writer.writerow(['closest_dropoff_b', 465, check_type,
                    shape_size])
        dist_matrix = torch_dijsktra(parameter_graph)
        error = loss(dist_matrix, closest_pickup_b, closest_dropoff_b, di_b,
            scaling, y_true, clip=1000000)
        if 'closest_pickup_b' not in TANGSHAN:
            import csv
            if isinstance(closest_pickup_b, np.ndarray) or isinstance(
                closest_pickup_b, pd.DataFrame) or isinstance(closest_pickup_b,
                pd.Series):
                shape_size = closest_pickup_b.shape
            elif isinstance(closest_pickup_b, list):
                shape_size = len(closest_pickup_b)
            else:
                shape_size = 'any'
            check_type = type(closest_pickup_b)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('closest_pickup_b')
                writer = csv.writer(f)
                writer.writerow(['closest_pickup_b', 467, check_type,
                    shape_size])
        np_loss = float(error.data.cpu().numpy())
        if 'float' not in TANGSHAN:
            import csv
            if isinstance(float, np.ndarray) or isinstance(float, pd.DataFrame
                ) or isinstance(float, pd.Series):
                shape_size = float.shape
            elif isinstance(float, list):
                shape_size = len(float)
            else:
                shape_size = 'any'
            check_type = type(float)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('float')
                writer = csv.writer(f)
                writer.writerow(['float', 468, check_type, shape_size])
        print('loss [epoch %i]' % epoch, np_loss)
        print('scaling', scaling.data.cpu().numpy())
print('optimization done')
G = REALLY_BIG_NUM * np.ones((dim, dim))
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
        writer.writerow(['dim', 476, check_type, shape_size])
for i, edges in enumerate(parameter_graph):
    for j, d, min_d in edges:
        G[i, j] = (d.clamp(min=0.9) * min_d).data.cpu().numpy()
        if 'G' not in TANGSHAN:
            import csv
            if isinstance(G, np.ndarray) or isinstance(G, pd.DataFrame
                ) or isinstance(G, pd.Series):
                shape_size = G.shape
            elif isinstance(G, list):
                shape_size = len(G)
            else:
                shape_size = 'any'
            check_type = type(G)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('G')
                writer = csv.writer(f)
                writer.writerow(['G', 479, check_type, shape_size])
    if 'min_d' not in TANGSHAN:
        import csv
        if isinstance(min_d, np.ndarray) or isinstance(min_d, pd.DataFrame
            ) or isinstance(min_d, pd.Series):
            shape_size = min_d.shape
        elif isinstance(min_d, list):
            shape_size = len(min_d)
        else:
            shape_size = 'any'
        check_type = type(min_d)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('min_d')
            writer = csv.writer(f)
            writer.writerow(['min_d', 478, check_type, shape_size])
if 'parameter_graph' not in TANGSHAN:
    import csv
    if isinstance(parameter_graph, np.ndarray) or isinstance(parameter_graph,
        pd.DataFrame) or isinstance(parameter_graph, pd.Series):
        shape_size = parameter_graph.shape
    elif isinstance(parameter_graph, list):
        shape_size = len(parameter_graph)
    else:
        shape_size = 'any'
    check_type = type(parameter_graph)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('parameter_graph')
        writer = csv.writer(f)
        writer.writerow(['parameter_graph', 477, check_type, shape_size])
save_graph(G, data_points, 'nyc_%f_a.png' % np_loss, 
    'absolute distances [loss: %f]' % np_loss, random_colors=False)
for i, edges in enumerate(parameter_graph):
    for j, d, min_d in edges:
        G[i, j] = d.clamp(min=0.9).data.cpu().numpy()
if 'edges' not in TANGSHAN:
    import csv
    if isinstance(edges, np.ndarray) or isinstance(edges, pd.DataFrame
        ) or isinstance(edges, pd.Series):
        shape_size = edges.shape
    elif isinstance(edges, list):
        shape_size = len(edges)
    else:
        shape_size = 'any'
    check_type = type(edges)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('edges')
        writer = csv.writer(f)
        writer.writerow(['edges', 483, check_type, shape_size])
save_graph(G, data_points, 'nyc_%f_b.png' % np_loss, 
    'distance adjustements [loss: %f]' % np_loss, random_colors=False)
if 'np_loss' not in TANGSHAN:
    import csv
    if isinstance(np_loss, np.ndarray) or isinstance(np_loss, pd.DataFrame
        ) or isinstance(np_loss, pd.Series):
        shape_size = np_loss.shape
    elif isinstance(np_loss, list):
        shape_size = len(np_loss)
    else:
        shape_size = 'any'
    check_type = type(np_loss)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('np_loss')
        writer = csv.writer(f)
        writer.writerow(['np_loss', 487, check_type, shape_size])
