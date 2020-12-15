import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from subprocess import check_output
TANGSHAN = []
#print(check_output(['ls', '../input']).decode('utf8'))
df = pd.read_csv(r'D:\dataset\big_10\bosch-production-line-performance\\train_numeric.csv', nrows=5000)
print('Total rows', df.count().get_value(label='Response'), '\n')
failedmean = df[df['Response'] == 1].mean(axis=0)
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
        writer.writerow(['df', 19, check_type, shape_size])
failedstd = df[df['Response'] == 1].std(axis=0)
if 'failedstd' not in TANGSHAN:
    import csv
    if isinstance(failedstd, np.ndarray) or isinstance(failedstd, pd.DataFrame
        ) or isinstance(failedstd, pd.Series):
        shape_size = failedstd.shape
    elif isinstance(failedstd, list):
        shape_size = len(failedstd)
    else:
        shape_size = 'any'
    check_type = type(failedstd)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('failedstd')
        writer = csv.writer(f)
        writer.writerow(['failedstd', 20, check_type, shape_size])
print('Failed std ', failedstd, '\n')
print('Failed std non NAN', df[df['Response'] == 1].count().sort_values(), '\n'
    )
s = df[df['Response'] == 1].count().sort_values()
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
        writer.writerow(['s', 23, check_type, shape_size])
s2 = s[s > 29]
print('Failed std non NAN gt 40', s2, '\n')
indexeswithoccurencegt40 = s2.head(s2.size - 2)
if 's2' not in TANGSHAN:
    import csv
    if isinstance(s2, np.ndarray) or isinstance(s2, pd.DataFrame
        ) or isinstance(s2, pd.Series):
        shape_size = s2.shape
    elif isinstance(s2, list):
        shape_size = len(s2)
    else:
        shape_size = 'any'
    check_type = type(s2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('s2')
        writer = csv.writer(f)
        writer.writerow(['s2', 26, check_type, shape_size])
df2 = pd.DataFrame(df, columns=indexeswithoccurencegt40.axes)
failedmeanwithoccurencegt40 = pd.Series(failedmean, index=
    indexeswithoccurencegt40.axes)
if 'failedmean' not in TANGSHAN:
    import csv
    if isinstance(failedmean, np.ndarray) or isinstance(failedmean, pd.
        DataFrame) or isinstance(failedmean, pd.Series):
        shape_size = failedmean.shape
    elif isinstance(failedmean, list):
        shape_size = len(failedmean)
    else:
        shape_size = 'any'
    check_type = type(failedmean)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('failedmean')
        writer = csv.writer(f)
        writer.writerow(['failedmean', 33, check_type, shape_size])
if 'failedmeanwithoccurencegt40' not in TANGSHAN:
    import csv
    if isinstance(failedmeanwithoccurencegt40, np.ndarray) or isinstance(
        failedmeanwithoccurencegt40, pd.DataFrame) or isinstance(
        failedmeanwithoccurencegt40, pd.Series):
        shape_size = failedmeanwithoccurencegt40.shape
    elif isinstance(failedmeanwithoccurencegt40, list):
        shape_size = len(failedmeanwithoccurencegt40)
    else:
        shape_size = 'any'
    check_type = type(failedmeanwithoccurencegt40)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('failedmeanwithoccurencegt40')
        writer = csv.writer(f)
        writer.writerow(['failedmeanwithoccurencegt40', 33, check_type,
            shape_size])
failedstdwithoccurencegt40 = pd.Series(failedstd, index=
    indexeswithoccurencegt40.axes)
if 'indexeswithoccurencegt40' not in TANGSHAN:
    import csv
    if isinstance(indexeswithoccurencegt40, np.ndarray) or isinstance(
        indexeswithoccurencegt40, pd.DataFrame) or isinstance(
        indexeswithoccurencegt40, pd.Series):
        shape_size = indexeswithoccurencegt40.shape
    elif isinstance(indexeswithoccurencegt40, list):
        shape_size = len(indexeswithoccurencegt40)
    else:
        shape_size = 'any'
    check_type = type(indexeswithoccurencegt40)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('indexeswithoccurencegt40')
        writer = csv.writer(f)
        writer.writerow(['indexeswithoccurencegt40', 34, check_type,
            shape_size])
minthreshold1 = np.inf
maxthreshold1 = -np.inf
minthreshold2 = np.inf
if 'minthreshold2' not in TANGSHAN:
    import csv
    if isinstance(minthreshold2, np.ndarray) or isinstance(minthreshold2,
        pd.DataFrame) or isinstance(minthreshold2, pd.Series):
        shape_size = minthreshold2.shape
    elif isinstance(minthreshold2, list):
        shape_size = len(minthreshold2)
    else:
        shape_size = 'any'
    check_type = type(minthreshold2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('minthreshold2')
        writer = csv.writer(f)
        writer.writerow(['minthreshold2', 37, check_type, shape_size])
maxthreshold2 = -np.inf
if 'maxthreshold2' not in TANGSHAN:
    import csv
    if isinstance(maxthreshold2, np.ndarray) or isinstance(maxthreshold2,
        pd.DataFrame) or isinstance(maxthreshold2, pd.Series):
        shape_size = maxthreshold2.shape
    elif isinstance(maxthreshold2, list):
        shape_size = len(maxthreshold2)
    else:
        shape_size = 'any'
    check_type = type(maxthreshold2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('maxthreshold2')
        writer = csv.writer(f)
        writer.writerow(['maxthreshold2', 38, check_type, shape_size])
for index, row in df2.iterrows():
    rowminusmean = row - failedmeanwithoccurencegt40
    if 'rowminusmean' not in TANGSHAN:
        import csv
        if isinstance(rowminusmean, np.ndarray) or isinstance(rowminusmean,
            pd.DataFrame) or isinstance(rowminusmean, pd.Series):
            shape_size = rowminusmean.shape
        elif isinstance(rowminusmean, list):
            shape_size = len(rowminusmean)
        else:
            shape_size = 'any'
        check_type = type(rowminusmean)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('rowminusmean')
            writer = csv.writer(f)
            writer.writerow(['rowminusmean', 40, check_type, shape_size])
    if 'row' not in TANGSHAN:
        import csv
        if isinstance(row, np.ndarray) or isinstance(row, pd.DataFrame
            ) or isinstance(row, pd.Series):
            shape_size = row.shape
        elif isinstance(row, list):
            shape_size = len(row)
        else:
            shape_size = 'any'
        check_type = type(row)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('row')
            writer = csv.writer(f)
            writer.writerow(['row', 40, check_type, shape_size])
    prob = np.exp(0.5 * -np.power(rowminusmean / failedstdwithoccurencegt40,
        2.0)) / (failedstdwithoccurencegt40 * np.sqrt(2 * math.pi))
    if 'failedstdwithoccurencegt40' not in TANGSHAN:
        import csv
        if isinstance(failedstdwithoccurencegt40, np.ndarray) or isinstance(
            failedstdwithoccurencegt40, pd.DataFrame) or isinstance(
            failedstdwithoccurencegt40, pd.Series):
            shape_size = failedstdwithoccurencegt40.shape
        elif isinstance(failedstdwithoccurencegt40, list):
            shape_size = len(failedstdwithoccurencegt40)
        else:
            shape_size = 'any'
        check_type = type(failedstdwithoccurencegt40)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('failedstdwithoccurencegt40')
            writer = csv.writer(f)
            writer.writerow(['failedstdwithoccurencegt40', 41, check_type,
                shape_size])
    CumprodProb = prob.prod()
    if df.iloc[index]['Response'] > 0.5:
        if CumprodProb < minthreshold1:
            minthreshold1 = CumprodProb
            if 'minthreshold1' not in TANGSHAN:
                import csv
                if isinstance(minthreshold1, np.ndarray) or isinstance(
                    minthreshold1, pd.DataFrame) or isinstance(minthreshold1,
                    pd.Series):
                    shape_size = minthreshold1.shape
                elif isinstance(minthreshold1, list):
                    shape_size = len(minthreshold1)
                else:
                    shape_size = 'any'
                check_type = type(minthreshold1)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('minthreshold1')
                    writer = csv.writer(f)
                    writer.writerow(['minthreshold1', 47, check_type,
                        shape_size])
        if CumprodProb > maxthreshold1:
            maxthreshold1 = CumprodProb
        if 'maxthreshold1' not in TANGSHAN:
            import csv
            if isinstance(maxthreshold1, np.ndarray) or isinstance(
                maxthreshold1, pd.DataFrame) or isinstance(maxthreshold1,
                pd.Series):
                shape_size = maxthreshold1.shape
            elif isinstance(maxthreshold1, list):
                shape_size = len(maxthreshold1)
            else:
                shape_size = 'any'
            check_type = type(maxthreshold1)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('maxthreshold1')
                writer = csv.writer(f)
                writer.writerow(['maxthreshold1', 48, check_type, shape_size])
    else:
        if CumprodProb < minthreshold2:
            minthreshold2 = CumprodProb
        if 'CumprodProb' not in TANGSHAN:
            import csv
            if isinstance(CumprodProb, np.ndarray) or isinstance(CumprodProb,
                pd.DataFrame) or isinstance(CumprodProb, pd.Series):
                shape_size = CumprodProb.shape
            elif isinstance(CumprodProb, list):
                shape_size = len(CumprodProb)
            else:
                shape_size = 'any'
            check_type = type(CumprodProb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('CumprodProb')
                writer = csv.writer(f)
                writer.writerow(['CumprodProb', 51, check_type, shape_size])
        if CumprodProb > maxthreshold2:
            maxthreshold2 = CumprodProb
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
            writer.writerow(['index', 45, check_type, shape_size])
if 'df2' not in TANGSHAN:
    import csv
    if isinstance(df2, np.ndarray) or isinstance(df2, pd.DataFrame
        ) or isinstance(df2, pd.Series):
        shape_size = df2.shape
    elif isinstance(df2, list):
        shape_size = len(df2)
    else:
        shape_size = 'any'
    check_type = type(df2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('df2')
        writer = csv.writer(f)
        writer.writerow(['df2', 39, check_type, shape_size])
print('minthreshold1 = ', minthreshold1)
print('maxthreshold1 = ', maxthreshold1)
print('minthreshold2 = ', minthreshold2)
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
        writer.writerow(['print', 57, check_type, shape_size])
print('maxthreshold2 = ', maxthreshold2)
plt.plot(prob.values)
if 'prob' not in TANGSHAN:
    import csv
    if isinstance(prob, np.ndarray) or isinstance(prob, pd.DataFrame
        ) or isinstance(prob, pd.Series):
        shape_size = prob.shape
    elif isinstance(prob, list):
        shape_size = len(prob)
    else:
        shape_size = 'any'
    check_type = type(prob)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('prob')
        writer = csv.writer(f)
        writer.writerow(['prob', 59, check_type, shape_size])
