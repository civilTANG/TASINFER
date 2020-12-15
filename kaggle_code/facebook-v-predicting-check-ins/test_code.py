import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(['ls', '../input']).decode('utf8'))
import os
if not os.path.exists('train/'):
    os.makedirs('train/')
if not os.path.exists('test/'):
    os.makedirs('test/')
if not os.path.exists('out/'):
    os.makedirs('out/')
print('starting...')
f = open('../input/train.csv', 'r')
header = f.readline()
total = 0
sub_region = dict()
length = 10.0
divisions = 10
block = length / divisions
overlap = 0.05
while 1:
    line = f.readline().strip()
    total += 1
    if total % 1000000 == 0:
        print(total)
    if line == '':
        break
    arr = line.split(',')
    row_id = arr[0]
    x = float(arr[1])
    y = float(arr[2])
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
            writer.writerow(['float', 54, check_type, shape_size])
    accuracy = arr[3]
    time = arr[4]
    if 'time' not in TANGSHAN:
        import csv
        if isinstance(time, np.ndarray) or isinstance(time, pd.DataFrame
            ) or isinstance(time, pd.Series):
            shape_size = time.shape
        elif isinstance(time, list):
            shape_size = len(time)
        else:
            shape_size = 'any'
        check_type = type(time)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('time')
            writer = csv.writer(f)
            writer.writerow(['time', 56, check_type, shape_size])
    place_id = arr[5]
    if 'place_id' not in TANGSHAN:
        import csv
        if isinstance(place_id, np.ndarray) or isinstance(place_id, pd.
            DataFrame) or isinstance(place_id, pd.Series):
            shape_size = place_id.shape
        elif isinstance(place_id, list):
            shape_size = len(place_id)
        else:
            shape_size = 'any'
        check_type = type(place_id)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('place_id')
            writer = csv.writer(f)
            writer.writerow(['place_id', 57, check_type, shape_size])
    if 'arr' not in TANGSHAN:
        import csv
        if isinstance(arr, np.ndarray) or isinstance(arr, pd.DataFrame
            ) or isinstance(arr, pd.Series):
            shape_size = arr.shape
        elif isinstance(arr, list):
            shape_size = len(arr)
        else:
            shape_size = 'any'
        check_type = type(arr)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('arr')
            writer = csv.writer(f)
            writer.writerow(['arr', 57, check_type, shape_size])
    istart = int(x / length * divisions)
    if 'istart' not in TANGSHAN:
        import csv
        if isinstance(istart, np.ndarray) or isinstance(istart, pd.DataFrame
            ) or isinstance(istart, pd.Series):
            shape_size = istart.shape
        elif isinstance(istart, list):
            shape_size = len(istart)
        else:
            shape_size = 'any'
        check_type = type(istart)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('istart')
            writer = csv.writer(f)
            writer.writerow(['istart', 60, check_type, shape_size])
    jstart = int(y / length * divisions)
    xstart = istart * block
    xend = xstart + block
    if xstart == 10.0:
        xstart = xstart - block
        xend = xstart + block
    ystart = jstart * block
    yend = ystart + block
    if ystart == 10.0:
        ystart = ystart - block
        yend = ystart + block
    key = str(xstart) + ' ' + str(xend) + ' ' + str(ystart) + ' ' + str(yend)
    if 'xend' not in TANGSHAN:
        import csv
        if isinstance(xend, np.ndarray) or isinstance(xend, pd.DataFrame
            ) or isinstance(xend, pd.Series):
            shape_size = xend.shape
        elif isinstance(xend, list):
            shape_size = len(xend)
        else:
            shape_size = 'any'
        check_type = type(xend)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('xend')
            writer = csv.writer(f)
            writer.writerow(['xend', 77, check_type, shape_size])
    if key not in sub_region.keys():
        sub_region[key] = []
        pass
    sub_region[key].append(line)
    if x <= xstart + overlap and xstart > 0.0:
        xstart2 = xstart - block
    else:
        if x >= xend - overlap and xend < 10.0:
            xstart2 = xstart + block
        else:
            xstart2 = -1
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
                writer.writerow(['x', 89, check_type, shape_size])
    if xstart2 != -1:
        xend2 = xstart2 + block
        if 'xstart2' not in TANGSHAN:
            import csv
            if isinstance(xstart2, np.ndarray) or isinstance(xstart2, pd.
                DataFrame) or isinstance(xstart2, pd.Series):
                shape_size = xstart2.shape
            elif isinstance(xstart2, list):
                shape_size = len(xstart2)
            else:
                shape_size = 'any'
            check_type = type(xstart2)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xstart2')
                writer = csv.writer(f)
                writer.writerow(['xstart2', 95, check_type, shape_size])
        keyx = str(xstart2) + ' ' + str(xend2) + ' ' + str(ystart) + ' ' + str(
            yend)
        if 'yend' not in TANGSHAN:
            import csv
            if isinstance(yend, np.ndarray) or isinstance(yend, pd.DataFrame
                ) or isinstance(yend, pd.Series):
                shape_size = yend.shape
            elif isinstance(yend, list):
                shape_size = len(yend)
            else:
                shape_size = 'any'
            check_type = type(yend)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('yend')
                writer = csv.writer(f)
                writer.writerow(['yend', 96, check_type, shape_size])
        if 'xend2' not in TANGSHAN:
            import csv
            if isinstance(xend2, np.ndarray) or isinstance(xend2, pd.DataFrame
                ) or isinstance(xend2, pd.Series):
                shape_size = xend2.shape
            elif isinstance(xend2, list):
                shape_size = len(xend2)
            else:
                shape_size = 'any'
            check_type = type(xend2)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xend2')
                writer = csv.writer(f)
                writer.writerow(['xend2', 96, check_type, shape_size])
        if keyx not in sub_region.keys():
            sub_region[keyx] = []
            pass
        if 'keyx' not in TANGSHAN:
            import csv
            if isinstance(keyx, np.ndarray) or isinstance(keyx, pd.DataFrame
                ) or isinstance(keyx, pd.Series):
                shape_size = keyx.shape
            elif isinstance(keyx, list):
                shape_size = len(keyx)
            else:
                shape_size = 'any'
            check_type = type(keyx)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('keyx')
                writer = csv.writer(f)
                writer.writerow(['keyx', 97, check_type, shape_size])
        sub_region[keyx].append(line)
        if 'sub_region' not in TANGSHAN:
            import csv
            if isinstance(sub_region, np.ndarray) or isinstance(sub_region,
                pd.DataFrame) or isinstance(sub_region, pd.Series):
                shape_size = sub_region.shape
            elif isinstance(sub_region, list):
                shape_size = len(sub_region)
            else:
                shape_size = 'any'
            check_type = type(sub_region)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('sub_region')
                writer = csv.writer(f)
                writer.writerow(['sub_region', 100, check_type, shape_size])
    if y <= ystart + overlap and ystart > 0.0:
        ystart2 = ystart - block
    elif y >= yend - overlap and yend < 10.0:
        ystart2 = ystart + block
        if 'ystart2' not in TANGSHAN:
            import csv
            if isinstance(ystart2, np.ndarray) or isinstance(ystart2, pd.
                DataFrame) or isinstance(ystart2, pd.Series):
                shape_size = ystart2.shape
            elif isinstance(ystart2, list):
                shape_size = len(ystart2)
            else:
                shape_size = 'any'
            check_type = type(ystart2)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('ystart2')
                writer = csv.writer(f)
                writer.writerow(['ystart2', 106, check_type, shape_size])
    else:
        ystart2 = -1
    if 'overlap' not in TANGSHAN:
        import csv
        if isinstance(overlap, np.ndarray) or isinstance(overlap, pd.DataFrame
            ) or isinstance(overlap, pd.Series):
            shape_size = overlap.shape
        elif isinstance(overlap, list):
            shape_size = len(overlap)
        else:
            shape_size = 'any'
        check_type = type(overlap)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('overlap')
            writer = csv.writer(f)
            writer.writerow(['overlap', 103, check_type, shape_size])
    if ystart2 != -1:
        yend2 = ystart2 + block
        keyy = str(xstart) + ' ' + str(xend) + ' ' + str(ystart2) + ' ' + str(
            yend2)
        if 'yend2' not in TANGSHAN:
            import csv
            if isinstance(yend2, np.ndarray) or isinstance(yend2, pd.DataFrame
                ) or isinstance(yend2, pd.Series):
                shape_size = yend2.shape
            elif isinstance(yend2, list):
                shape_size = len(yend2)
            else:
                shape_size = 'any'
            check_type = type(yend2)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('yend2')
                writer = csv.writer(f)
                writer.writerow(['yend2', 112, check_type, shape_size])
        if keyy not in sub_region.keys():
            sub_region[keyy] = []
            pass
        if 'keyy' not in TANGSHAN:
            import csv
            if isinstance(keyy, np.ndarray) or isinstance(keyy, pd.DataFrame
                ) or isinstance(keyy, pd.Series):
                shape_size = keyy.shape
            elif isinstance(keyy, list):
                shape_size = len(keyy)
            else:
                shape_size = 'any'
            check_type = type(keyy)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('keyy')
                writer = csv.writer(f)
                writer.writerow(['keyy', 113, check_type, shape_size])
        sub_region[keyy].append(line)
    if x <= xstart + overlap and xstart > 0.0:
        xstart3 = xstart - block
        if 'xstart3' not in TANGSHAN:
            import csv
            if isinstance(xstart3, np.ndarray) or isinstance(xstart3, pd.
                DataFrame) or isinstance(xstart3, pd.Series):
                shape_size = xstart3.shape
            elif isinstance(xstart3, list):
                shape_size = len(xstart3)
            else:
                shape_size = 'any'
            check_type = type(xstart3)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xstart3')
                writer = csv.writer(f)
                writer.writerow(['xstart3', 120, check_type, shape_size])
        if y <= ystart + overlap and ystart > 0.0:
            ystart3 = ystart - block
        elif y >= yend - overlap and yend < 10.0:
            ystart3 = ystart + block
            if 'block' not in TANGSHAN:
                import csv
                if isinstance(block, np.ndarray) or isinstance(block, pd.
                    DataFrame) or isinstance(block, pd.Series):
                    shape_size = block.shape
                elif isinstance(block, list):
                    shape_size = len(block)
                else:
                    shape_size = 'any'
                check_type = type(block)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('block')
                    writer = csv.writer(f)
                    writer.writerow(['block', 124, check_type, shape_size])
        else:
            ystart3 = -1
    elif x >= xend - overlap and xend < 10.0:
        xstart3 = xstart + block
        if y <= ystart + overlap and ystart > 0.0:
            ystart3 = ystart - block
        else:
            if y >= yend - overlap and yend < 10.0:
                ystart3 = ystart + block
            else:
                ystart3 = -1
                if 'ystart3' not in TANGSHAN:
                    import csv
                    if isinstance(ystart3, np.ndarray) or isinstance(ystart3,
                        pd.DataFrame) or isinstance(ystart3, pd.Series):
                        shape_size = ystart3.shape
                    elif isinstance(ystart3, list):
                        shape_size = len(ystart3)
                    else:
                        shape_size = 'any'
                    check_type = type(ystart3)
                    with open('tas.csv', 'a+') as f:
                        TANGSHAN.append('ystart3')
                        writer = csv.writer(f)
                        writer.writerow(['ystart3', 134, check_type,
                            shape_size])
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
                    writer.writerow(['y', 131, check_type, shape_size])
    else:
        xstart3 = -1
        ystart3 = -1
    if ystart3 != -1:
        xend3 = xstart3 + block
        yend3 = ystart3 + block
        keyxy = str(xstart3) + ' ' + str(xend3) + ' ' + str(ystart3
            ) + ' ' + str(yend3)
        if 'yend3' not in TANGSHAN:
            import csv
            if isinstance(yend3, np.ndarray) or isinstance(yend3, pd.DataFrame
                ) or isinstance(yend3, pd.Series):
                shape_size = yend3.shape
            elif isinstance(yend3, list):
                shape_size = len(yend3)
            else:
                shape_size = 'any'
            check_type = type(yend3)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('yend3')
                writer = csv.writer(f)
                writer.writerow(['yend3', 142, check_type, shape_size])
        if 'xend3' not in TANGSHAN:
            import csv
            if isinstance(xend3, np.ndarray) or isinstance(xend3, pd.DataFrame
                ) or isinstance(xend3, pd.Series):
                shape_size = xend3.shape
            elif isinstance(xend3, list):
                shape_size = len(xend3)
            else:
                shape_size = 'any'
            check_type = type(xend3)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xend3')
                writer = csv.writer(f)
                writer.writerow(['xend3', 142, check_type, shape_size])
        if keyxy not in sub_region.keys():
            sub_region[keyxy] = []
            pass
        sub_region[keyxy].append(line)
        if 'keyxy' not in TANGSHAN:
            import csv
            if isinstance(keyxy, np.ndarray) or isinstance(keyxy, pd.DataFrame
                ) or isinstance(keyxy, pd.Series):
                shape_size = keyxy.shape
            elif isinstance(keyxy, list):
                shape_size = len(keyxy)
            else:
                shape_size = 'any'
            check_type = type(keyxy)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('keyxy')
                writer = csv.writer(f)
                writer.writerow(['keyxy', 146, check_type, shape_size])
f.close()
print('writing...')
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
        writer.writerow(['print', 150, check_type, shape_size])
total = 0
for key in sub_region.keys():
    total += 1
    if total % 10 == 0:
        print(total)
    f = open('train/' + key + '.csv', 'w')
    f.write(header + '\n')
    if 'header' not in TANGSHAN:
        import csv
        if isinstance(header, np.ndarray) or isinstance(header, pd.DataFrame
            ) or isinstance(header, pd.Series):
            shape_size = header.shape
        elif isinstance(header, list):
            shape_size = len(header)
        else:
            shape_size = 'any'
        check_type = type(header)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('header')
            writer = csv.writer(f)
            writer.writerow(['header', 158, check_type, shape_size])
    for line in sub_region[key]:
        f.write(line + '\n')
    if 'key' not in TANGSHAN:
        import csv
        if isinstance(key, np.ndarray) or isinstance(key, pd.DataFrame
            ) or isinstance(key, pd.Series):
            shape_size = key.shape
        elif isinstance(key, list):
            shape_size = len(key)
        else:
            shape_size = 'any'
        check_type = type(key)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('key')
            writer = csv.writer(f)
            writer.writerow(['key', 159, check_type, shape_size])
    f.close()
print('done...')
print('starting...')
f = open('../input/test.csv', 'r')
header = f.readline()
total = 0
sub_region = dict()
if 'dict' not in TANGSHAN:
    import csv
    if isinstance(dict, np.ndarray) or isinstance(dict, pd.DataFrame
        ) or isinstance(dict, pd.Series):
        shape_size = dict.shape
    elif isinstance(dict, list):
        shape_size = len(dict)
    else:
        shape_size = 'any'
    check_type = type(dict)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dict')
        writer = csv.writer(f)
        writer.writerow(['dict', 172, check_type, shape_size])
length = 10.0
if 'length' not in TANGSHAN:
    import csv
    if isinstance(length, np.ndarray) or isinstance(length, pd.DataFrame
        ) or isinstance(length, pd.Series):
        shape_size = length.shape
    elif isinstance(length, list):
        shape_size = len(length)
    else:
        shape_size = 'any'
    check_type = type(length)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('length')
        writer = csv.writer(f)
        writer.writerow(['length', 173, check_type, shape_size])
divisions = 10
block = length / divisions
overlap = 0.05
while 1:
    line = f.readline().strip()
    total += 1
    if total % 100000 == 0:
        print(total)
    if line == '':
        break
    arr = line.split(',')
    row_id = arr[0]
    if 'row_id' not in TANGSHAN:
        import csv
        if isinstance(row_id, np.ndarray) or isinstance(row_id, pd.DataFrame
            ) or isinstance(row_id, pd.Series):
            shape_size = row_id.shape
        elif isinstance(row_id, list):
            shape_size = len(row_id)
        else:
            shape_size = 'any'
        check_type = type(row_id)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('row_id')
            writer = csv.writer(f)
            writer.writerow(['row_id', 190, check_type, shape_size])
    x = float(arr[1])
    y = float(arr[2])
    accuracy = arr[3]
    if 'accuracy' not in TANGSHAN:
        import csv
        if isinstance(accuracy, np.ndarray) or isinstance(accuracy, pd.
            DataFrame) or isinstance(accuracy, pd.Series):
            shape_size = accuracy.shape
        elif isinstance(accuracy, list):
            shape_size = len(accuracy)
        else:
            shape_size = 'any'
        check_type = type(accuracy)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('accuracy')
            writer = csv.writer(f)
            writer.writerow(['accuracy', 193, check_type, shape_size])
    time = arr[4]
    istart = int(x / length * divisions)
    jstart = int(y / length * divisions)
    if 'jstart' not in TANGSHAN:
        import csv
        if isinstance(jstart, np.ndarray) or isinstance(jstart, pd.DataFrame
            ) or isinstance(jstart, pd.Series):
            shape_size = jstart.shape
        elif isinstance(jstart, list):
            shape_size = len(jstart)
        else:
            shape_size = 'any'
        check_type = type(jstart)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('jstart')
            writer = csv.writer(f)
            writer.writerow(['jstart', 199, check_type, shape_size])
    xstart = istart * block
    xend = xstart + block
    if xstart == 10.0:
        xstart = xstart - block
        xend = xstart + block
    ystart = jstart * block
    yend = ystart + block
    if ystart == 10.0:
        ystart = ystart - block
        yend = ystart + block
    key = str(xstart) + ' ' + str(xend) + ' ' + str(ystart) + ' ' + str(yend)
    if key not in sub_region.keys():
        sub_region[key] = []
        pass
    sub_region[key].append(line)
    if 'line' not in TANGSHAN:
        import csv
        if isinstance(line, np.ndarray) or isinstance(line, pd.DataFrame
            ) or isinstance(line, pd.Series):
            shape_size = line.shape
        elif isinstance(line, list):
            shape_size = len(line)
        else:
            shape_size = 'any'
        check_type = type(line)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('line')
            writer = csv.writer(f)
            writer.writerow(['line', 221, check_type, shape_size])
f.close()
print('writing...')
total = 0
for key in sub_region.keys():
    total += 1
    if total % 10 == 0:
        print(total)
    if 'total' not in TANGSHAN:
        import csv
        if isinstance(total, np.ndarray) or isinstance(total, pd.DataFrame
            ) or isinstance(total, pd.Series):
            shape_size = total.shape
        elif isinstance(total, list):
            shape_size = len(total)
        else:
            shape_size = 'any'
        check_type = type(total)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('total')
            writer = csv.writer(f)
            writer.writerow(['total', 230, check_type, shape_size])
    f = open('test/' + key + '.csv', 'w')
    f.write(header + '\n')
    if 'f' not in TANGSHAN:
        import csv
        if isinstance(f, np.ndarray) or isinstance(f, pd.DataFrame
            ) or isinstance(f, pd.Series):
            shape_size = f.shape
        elif isinstance(f, list):
            shape_size = len(f)
        else:
            shape_size = 'any'
        check_type = type(f)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('f')
            writer = csv.writer(f)
            writer.writerow(['f', 233, check_type, shape_size])
    for line in sub_region[key]:
        f.write(line + '\n')
    f.close()
print('done...')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
length = 10.0
divisions = 10
block = length / divisions
if 'divisions' not in TANGSHAN:
    import csv
    if isinstance(divisions, np.ndarray) or isinstance(divisions, pd.DataFrame
        ) or isinstance(divisions, pd.Series):
        shape_size = divisions.shape
    elif isinstance(divisions, list):
        shape_size = len(divisions)
    else:
        shape_size = 'any'
    check_type = type(divisions)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('divisions')
        writer = csv.writer(f)
        writer.writerow(['divisions', 249, check_type, shape_size])
for i in range(0, divisions):
    xstart = i * block
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
            writer.writerow(['i', 252, check_type, shape_size])
    xend = xstart + block
    for j in range(0, divisions):
        ystart = j * block
        yend = ystart + block
        key = str(xstart) + ' ' + str(xend) + ' ' + str(ystart) + ' ' + str(
            yend)
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
                writer.writerow(['str', 258, check_type, shape_size])
        train_df = pd.read_csv('train/' + key + '.csv', encoding=
            'ISO-8859-1', header=0)
        if 'train_df' not in TANGSHAN:
            import csv
            if isinstance(train_df, np.ndarray) or isinstance(train_df, pd.
                DataFrame) or isinstance(train_df, pd.Series):
                shape_size = train_df.shape
            elif isinstance(train_df, list):
                shape_size = len(train_df)
            else:
                shape_size = 'any'
            check_type = type(train_df)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('train_df')
                writer = csv.writer(f)
                writer.writerow(['train_df', 260, check_type, shape_size])
        test_df = pd.read_csv('test/' + key + '.csv', encoding='ISO-8859-1',
            header=0)
        initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn),
            'm') for mn in train_df.time.values)
        if 'initial_date' not in TANGSHAN:
            import csv
            if isinstance(initial_date, np.ndarray) or isinstance(initial_date,
                pd.DataFrame) or isinstance(initial_date, pd.Series):
                shape_size = initial_date.shape
            elif isinstance(initial_date, list):
                shape_size = len(initial_date)
            else:
                shape_size = 'any'
            check_type = type(initial_date)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('initial_date')
                writer = csv.writer(f)
                writer.writerow(['initial_date', 265, check_type, shape_size])
        if 'mn' not in TANGSHAN:
            import csv
            if isinstance(mn, np.ndarray) or isinstance(mn, pd.DataFrame
                ) or isinstance(mn, pd.Series):
                shape_size = mn.shape
            elif isinstance(mn, list):
                shape_size = len(mn)
            else:
                shape_size = 'any'
            check_type = type(mn)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('mn')
                writer = csv.writer(f)
                writer.writerow(['mn', 265, check_type, shape_size])
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
                writer.writerow(['int', 265, check_type, shape_size])
        train_df['hour'] = d_times.hour
        train_df['weekday'] = d_times.weekday
        train_df['day'] = d_times.day
        train_df['month'] = d_times.month
        train_df['year'] = d_times.year
        train_df['logacc'] = np.log(train_df.accuracy)
        train_df = train_df.drop(['time'], axis=1)
        train_df = train_df.groupby('place_id').filter(lambda x: len(x) > 500)
        d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn),
            'm') for mn in test_df.time.values)
        test_df['hour'] = d_times.hour
        test_df['weekday'] = d_times.weekday
        test_df['day'] = d_times.day
        test_df['month'] = d_times.month
        if 'd_times' not in TANGSHAN:
            import csv
            if isinstance(d_times, np.ndarray) or isinstance(d_times, pd.
                DataFrame) or isinstance(d_times, pd.Series):
                shape_size = d_times.shape
            elif isinstance(d_times, list):
                shape_size = len(d_times)
            else:
                shape_size = 'any'
            check_type = type(d_times)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('d_times')
                writer = csv.writer(f)
                writer.writerow(['d_times', 281, check_type, shape_size])
        test_df['year'] = d_times.year
        test_df['logacc'] = np.log(test_df.accuracy)
        test_df = test_df.drop(['time'], axis=1)
        if 'test_df' not in TANGSHAN:
            import csv
            if isinstance(test_df, np.ndarray) or isinstance(test_df, pd.
                DataFrame) or isinstance(test_df, pd.Series):
                shape_size = test_df.shape
            elif isinstance(test_df, list):
                shape_size = len(test_df)
            else:
                shape_size = 'any'
            check_type = type(test_df)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('test_df')
                writer = csv.writer(f)
                writer.writerow(['test_df', 284, check_type, shape_size])
        le = preprocessing.LabelEncoder()
        le.fit(train_df['place_id'].as_matrix())
        train_Y = le.transform(train_df['place_id'].as_matrix())
        if 'train_Y' not in TANGSHAN:
            import csv
            if isinstance(train_Y, np.ndarray) or isinstance(train_Y, pd.
                DataFrame) or isinstance(train_Y, pd.Series):
                shape_size = train_Y.shape
            elif isinstance(train_Y, list):
                shape_size = len(train_Y)
            else:
                shape_size = 'any'
            check_type = type(train_Y)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('train_Y')
                writer = csv.writer(f)
                writer.writerow(['train_Y', 289, check_type, shape_size])
        train_X = train_df.drop('place_id', 1).drop('row_id', 1).as_matrix()
        if 'train_X' not in TANGSHAN:
            import csv
            if isinstance(train_X, np.ndarray) or isinstance(train_X, pd.
                DataFrame) or isinstance(train_X, pd.Series):
                shape_size = train_X.shape
            elif isinstance(train_X, list):
                shape_size = len(train_X)
            else:
                shape_size = 'any'
            check_type = type(train_X)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('train_X')
                writer = csv.writer(f)
                writer.writerow(['train_X', 292, check_type, shape_size])
        test_X = test_df.drop('row_id', 1).as_matrix()
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X)
        if 'test_X' not in TANGSHAN:
            import csv
            if isinstance(test_X, np.ndarray) or isinstance(test_X, pd.
                DataFrame) or isinstance(test_X, pd.Series):
                shape_size = test_X.shape
            elif isinstance(test_X, list):
                shape_size = len(test_X)
            else:
                shape_size = 'any'
            check_type = type(test_X)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('test_X')
                writer = csv.writer(f)
                writer.writerow(['test_X', 296, check_type, shape_size])
        if 'xg_test' not in TANGSHAN:
            import csv
            if isinstance(xg_test, np.ndarray) or isinstance(xg_test, pd.
                DataFrame) or isinstance(xg_test, pd.Series):
                shape_size = xg_test.shape
            elif isinstance(xg_test, list):
                shape_size = len(xg_test)
            else:
                shape_size = 'any'
            check_type = type(xg_test)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xg_test')
                writer = csv.writer(f)
                writer.writerow(['xg_test', 296, check_type, shape_size])
        param = {}
        param['objective'] = 'multi:softprob'
        if 'param' not in TANGSHAN:
            import csv
            if isinstance(param, np.ndarray) or isinstance(param, pd.DataFrame
                ) or isinstance(param, pd.Series):
                shape_size = param.shape
            elif isinstance(param, list):
                shape_size = len(param)
            else:
                shape_size = 'any'
            check_type = type(param)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('param')
                writer = csv.writer(f)
                writer.writerow(['param', 301, check_type, shape_size])
        param['eta'] = 0.2
        param['max_depth'] = 10
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = len(train_df['place_id'].unique())
        watchlist = [(xg_train, 'train')]
        num_round = 10
        bst = xgb.train(param, xg_train, num_round, watchlist)
        if 'bst' not in TANGSHAN:
            import csv
            if isinstance(bst, np.ndarray) or isinstance(bst, pd.DataFrame
                ) or isinstance(bst, pd.Series):
                shape_size = bst.shape
            elif isinstance(bst, list):
                shape_size = len(bst)
            else:
                shape_size = 'any'
            check_type = type(bst)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('bst')
                writer = csv.writer(f)
                writer.writerow(['bst', 312, check_type, shape_size])
        if 'watchlist' not in TANGSHAN:
            import csv
            if isinstance(watchlist, np.ndarray) or isinstance(watchlist,
                pd.DataFrame) or isinstance(watchlist, pd.Series):
                shape_size = watchlist.shape
            elif isinstance(watchlist, list):
                shape_size = len(watchlist)
            else:
                shape_size = 'any'
            check_type = type(watchlist)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('watchlist')
                writer = csv.writer(f)
                writer.writerow(['watchlist', 312, check_type, shape_size])
        if 'xg_train' not in TANGSHAN:
            import csv
            if isinstance(xg_train, np.ndarray) or isinstance(xg_train, pd.
                DataFrame) or isinstance(xg_train, pd.Series):
                shape_size = xg_train.shape
            elif isinstance(xg_train, list):
                shape_size = len(xg_train)
            else:
                shape_size = 'any'
            check_type = type(xg_train)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xg_train')
                writer = csv.writer(f)
                writer.writerow(['xg_train', 312, check_type, shape_size])
        if 'num_round' not in TANGSHAN:
            import csv
            if isinstance(num_round, np.ndarray) or isinstance(num_round,
                pd.DataFrame) or isinstance(num_round, pd.Series):
                shape_size = num_round.shape
            elif isinstance(num_round, list):
                shape_size = len(num_round)
            else:
                shape_size = 'any'
            check_type = type(num_round)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('num_round')
                writer = csv.writer(f)
                writer.writerow(['num_round', 312, check_type, shape_size])
        pred_raw = bst.predict(xg_test)
        if 'pred_raw' not in TANGSHAN:
            import csv
            if isinstance(pred_raw, np.ndarray) or isinstance(pred_raw, pd.
                DataFrame) or isinstance(pred_raw, pd.Series):
                shape_size = pred_raw.shape
            elif isinstance(pred_raw, list):
                shape_size = len(pred_raw)
            else:
                shape_size = 'any'
            check_type = type(pred_raw)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('pred_raw')
                writer = csv.writer(f)
                writer.writerow(['pred_raw', 315, check_type, shape_size])

        def get_top3_places_and_probs(row):
            row.sort_values(inplace=True)
            inds = row.index[-3:][::-1].tolist()
            return inds
        result_xgb_df = pd.DataFrame(index=test_df.row_id, data=pred_raw)
        result_xgb_df['pred'] = result_xgb_df.apply(get_top3_places_and_probs,
            axis=1)
        result_xgb_df['pred_0'] = result_xgb_df['pred'].map(lambda x: x[0])
        result_xgb_df['pred_1'] = result_xgb_df['pred'].map(lambda x: x[1])
        result_xgb_df['pred_2'] = result_xgb_df['pred'].map(lambda x: x[2])
        if 'result_xgb_df' not in TANGSHAN:
            import csv
            if isinstance(result_xgb_df, np.ndarray) or isinstance(
                result_xgb_df, pd.DataFrame) or isinstance(result_xgb_df,
                pd.Series):
                shape_size = result_xgb_df.shape
            elif isinstance(result_xgb_df, list):
                shape_size = len(result_xgb_df)
            else:
                shape_size = 'any'
            check_type = type(result_xgb_df)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('result_xgb_df')
                writer = csv.writer(f)
                writer.writerow(['result_xgb_df', 326, check_type, shape_size])
        result_xgb_df['pred_0'] = result_xgb_df['pred_0'].apply(le.
            inverse_transform)
        if 'le' not in TANGSHAN:
            import csv
            if isinstance(le, np.ndarray) or isinstance(le, pd.DataFrame
                ) or isinstance(le, pd.Series):
                shape_size = le.shape
            elif isinstance(le, list):
                shape_size = len(le)
            else:
                shape_size = 'any'
            check_type = type(le)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('le')
                writer = csv.writer(f)
                writer.writerow(['le', 328, check_type, shape_size])
        result_xgb_df['pred_1'] = result_xgb_df['pred_1'].apply(le.
            inverse_transform)
        result_xgb_df['pred_2'] = result_xgb_df['pred_2'].apply(le.
            inverse_transform)
        result_xgb_df['place_id'] = result_xgb_df['pred_0'].map(str
            ) + ' ' + result_xgb_df['pred_1'].map(str) + ' ' + result_xgb_df[
            'pred_2'].map(str)
        submit = result_xgb_df[['place_id']]
        submit.to_csv('out/' + key + '.csv')
        if 'submit' not in TANGSHAN:
            import csv
            if isinstance(submit, np.ndarray) or isinstance(submit, pd.
                DataFrame) or isinstance(submit, pd.Series):
                shape_size = submit.shape
            elif isinstance(submit, list):
                shape_size = len(submit)
            else:
                shape_size = 'any'
            check_type = type(submit)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('submit')
                writer = csv.writer(f)
                writer.writerow(['submit', 334, check_type, shape_size])
        print(key)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
length = 10.0
divisions = 10
block = length / divisions
df_all = None
for i in range(0, divisions):
    xstart = i * block
    xend = xstart + block
    for j in range(0, divisions):
        ystart = j * block
        yend = ystart + block
        if 'ystart' not in TANGSHAN:
            import csv
            if isinstance(ystart, np.ndarray) or isinstance(ystart, pd.
                DataFrame) or isinstance(ystart, pd.Series):
                shape_size = ystart.shape
            elif isinstance(ystart, list):
                shape_size = len(ystart)
            else:
                shape_size = 'any'
            check_type = type(ystart)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('ystart')
                writer = csv.writer(f)
                writer.writerow(['ystart', 354, check_type, shape_size])
        key = str(xstart) + ' ' + str(xend) + ' ' + str(ystart) + ' ' + str(
            yend)
        if 'xstart' not in TANGSHAN:
            import csv
            if isinstance(xstart, np.ndarray) or isinstance(xstart, pd.
                DataFrame) or isinstance(xstart, pd.Series):
                shape_size = xstart.shape
            elif isinstance(xstart, list):
                shape_size = len(xstart)
            else:
                shape_size = 'any'
            check_type = type(xstart)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('xstart')
                writer = csv.writer(f)
                writer.writerow(['xstart', 356, check_type, shape_size])
        df = pd.read_csv('out/' + key + '.csv', encoding='ISO-8859-1', header=0
            )
        if df_all is None:
            df_all = df
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
                    writer.writerow(['df', 361, check_type, shape_size])
        else:
            df_all = df_all.append(df)
        if 'df_all' not in TANGSHAN:
            import csv
            if isinstance(df_all, np.ndarray) or isinstance(df_all, pd.
                DataFrame) or isinstance(df_all, pd.Series):
                shape_size = df_all.shape
            elif isinstance(df_all, list):
                shape_size = len(df_all)
            else:
                shape_size = 'any'
            check_type = type(df_all)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('df_all')
                writer = csv.writer(f)
                writer.writerow(['df_all', 360, check_type, shape_size])
        print(key)
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
            writer.writerow(['range', 352, check_type, shape_size])
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
            writer.writerow(['j', 352, check_type, shape_size])
df_all.to_csv('out.csv', index=False)
