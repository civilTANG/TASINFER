import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
type_mapping = {'Ghoul': 0, 'Goblin': 1, 'Ghost': 2}
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
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('list')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['list', 11, check_type, shape_size])
if 'type_mapping' not in TANGSHAN:
    import csv
    if isinstance(type_mapping, np.ndarray) or isinstance(type_mapping, pd.
        DataFrame) or isinstance(type_mapping, pd.Series):
        shape_size = type_mapping.shape
    elif isinstance(type_mapping, list):
        shape_size = len(type_mapping)
    else:
        shape_size = 'any'
    check_type = type(type_mapping)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('type_mapping')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
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
                writer.writerow(['test_df', 25, check_type, shape_size])
        writer.writerow(['type_mapping', 11, check_type, shape_size])
reverse_type_mapping = {(0): 'Ghoul', (1): 'Goblin', (2): 'Ghost'}
train_df['type'] = train_df['type'].copy().map(type_mapping)
comb = list(itertools.combinations(train_df.drop(['id', 'color', 'type'],
    axis=1).columns, 2))
try_comb = pd.DataFrame()
for c in comb:
    try_comb[c[0] + '_x_' + c[1]] = train_df[c[0]].values * train_df[c[1]
        ].values
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['c', 19, check_type, shape_size])
try_comb['type'] = train_df.type
if 'try_comb' not in TANGSHAN:
    import csv
    if isinstance(try_comb, np.ndarray) or isinstance(try_comb, pd.DataFrame
        ) or isinstance(try_comb, pd.Series):
        shape_size = try_comb.shape
    elif isinstance(try_comb, list):
        shape_size = len(try_comb)
    else:
        shape_size = 'any'
    check_type = type(try_comb)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('try_comb')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['try_comb', 21, check_type, shape_size])
for i in [1, 2, -1]:
    train_df[comb[i][0] + '_x_' + comb[i][1]] = train_df[comb[i][0]
        ].values * train_df[comb[i][1]].values
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
            writer.writerow(['train_df', 24, check_type, shape_size])
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
            writer.writerow(['i', 24, check_type, shape_size])
    if 'comb' not in TANGSHAN:
        import csv
        if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
            ) or isinstance(comb, pd.Series):
            shape_size = comb.shape
        elif isinstance(comb, list):
            shape_size = len(comb)
        else:
            shape_size = 'any'
        check_type = type(comb)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('comb')
            writer = csv.writer(f)
            writer.writerow(['comb', 24, check_type, shape_size])
    test_df[comb[i][0] + '_x_' + comb[i][1]] = test_df[comb[i][0]
        ].values * test_df[comb[i][1]].values
    if 'test_df' not in TANGSHAN:
        import csv
        if isinstance(test_df, np.ndarray) or isinstance(test_df, pd.DataFrame
            ) or isinstance(test_df, pd.Series):
            shape_size = test_df.shape
        elif isinstance(test_df, list):
            shape_size = len(test_df)
        else:
            shape_size = 'any'
        check_type = type(test_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('test_df')
            writer = csv.writer(f)
            writer.writerow(['test_df', 25, check_type, shape_size])
train_df['hair_x_soul_x_bone'] = train_df['hair_length'].values * train_df[
    'has_soul'].values * train_df['bone_length'].values
test_df['hair_x_soul_x_bone'] = test_df['hair_length'].values * test_df[
    'has_soul'].values * test_df['bone_length'].values
labels_df = train_df['type'].copy()
features_df = train_df.copy().drop(['id', 'type', 'color'], axis=1)
if 'features_df' not in TANGSHAN:
    import csv
    if isinstance(features_df, np.ndarray) or isinstance(features_df, pd.
        DataFrame) or isinstance(features_df, pd.Series):
        shape_size = features_df.shape
    elif isinstance(features_df, list):
        shape_size = len(features_df)
    else:
        shape_size = 'any'
    check_type = type(features_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('features_df')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['features_df', 32, check_type, shape_size])
test_ids = test_df['id'].copy()
if 'test_ids' not in TANGSHAN:
    import csv
    if isinstance(test_ids, np.ndarray) or isinstance(test_ids, pd.DataFrame
        ) or isinstance(test_ids, pd.Series):
        shape_size = test_ids.shape
    elif isinstance(test_ids, list):
        shape_size = len(test_ids)
    else:
        shape_size = 'any'
    check_type = type(test_ids)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test_ids')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['test_ids', 34, check_type, shape_size])
test_features_df = test_df.copy().drop(['id', 'color'], axis=1)
all_features = pd.concat([features_df, test_features_df])
all_labels = pd.concat([labels_df, pd.DataFrame([-1] * len(test_features_df))])
if 'all_labels' not in TANGSHAN:
    import csv
    if isinstance(all_labels, np.ndarray) or isinstance(all_labels, pd.
        DataFrame) or isinstance(all_labels, pd.Series):
        shape_size = all_labels.shape
    elif isinstance(all_labels, list):
        shape_size = len(all_labels)
    else:
        shape_size = 'any'
    check_type = type(all_labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('all_labels')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['all_labels', 38, check_type, shape_size])
NR_FOLDS = 5
skf = StratifiedKFold(labels_df.values, n_folds=NR_FOLDS, shuffle=True,
    random_state=1)
if 'NR_FOLDS' not in TANGSHAN:
    import csv
    if isinstance(NR_FOLDS, np.ndarray) or isinstance(NR_FOLDS, pd.DataFrame
        ) or isinstance(NR_FOLDS, pd.Series):
        shape_size = NR_FOLDS.shape
    elif isinstance(NR_FOLDS, list):
        shape_size = len(NR_FOLDS)
    else:
        shape_size = 'any'
    check_type = type(NR_FOLDS)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('NR_FOLDS')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['NR_FOLDS', 41, check_type, shape_size])
if 'skf' not in TANGSHAN:
    import csv
    if isinstance(skf, np.ndarray) or isinstance(skf, pd.DataFrame
        ) or isinstance(skf, pd.Series):
        shape_size = skf.shape
    elif isinstance(skf, list):
        shape_size = len(skf)
    else:
        shape_size = 'any'
    check_type = type(skf)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('skf')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['skf', 41, check_type, shape_size])
accuracies = []
for fold, (train_idx, test_idx) in enumerate(skf):
    X_train = features_df.iloc[(train_idx), :].reset_index(drop=True)
    y_train = labels_df.iloc[train_idx].reset_index(drop=True)
    if 'train_idx' not in TANGSHAN:
        import csv
        if isinstance(train_idx, np.ndarray) or isinstance(train_idx, pd.
            DataFrame) or isinstance(train_idx, pd.Series):
            shape_size = train_idx.shape
        elif isinstance(train_idx, list):
            shape_size = len(train_idx)
        else:
            shape_size = 'any'
        check_type = type(train_idx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train_idx')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['train_idx', 45, check_type, shape_size])
    X_test = features_df.iloc[(test_idx), :].reset_index(drop=True)
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['X_test', 46, check_type, shape_size])
    y_test = labels_df.iloc[test_idx].reset_index(drop=True)
    if 'labels_df' not in TANGSHAN:
        import csv
        if isinstance(labels_df, np.ndarray) or isinstance(labels_df, pd.
            DataFrame) or isinstance(labels_df, pd.Series):
            shape_size = labels_df.shape
        elif isinstance(labels_df, list):
            shape_size = len(labels_df)
        else:
            shape_size = 'any'
        check_type = type(labels_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('labels_df')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['labels_df', 47, check_type, shape_size])
    if 'test_idx' not in TANGSHAN:
        import csv
        if isinstance(test_idx, np.ndarray) or isinstance(test_idx, pd.
            DataFrame) or isinstance(test_idx, pd.Series):
            shape_size = test_idx.shape
        elif isinstance(test_idx, list):
            shape_size = len(test_idx)
        else:
            shape_size = 'any'
        check_type = type(test_idx)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('test_idx')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            if 'test_df' not in TANGSHAN:
                import csv
                if isinstance(test_df, np.ndarray) or isinstance(test_df,
                    pd.DataFrame) or isinstance(test_df, pd.Series):
                    shape_size = test_df.shape
                elif isinstance(test_df, list):
                    shape_size = len(test_df)
                else:
                    shape_size = 'any'
                check_type = type(test_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('test_df')
                    writer = csv.writer(f)
                    writer.writerow(['test_df', 25, check_type, shape_size])
            writer.writerow(['test_idx', 47, check_type, shape_size])
    if 'y_test' not in TANGSHAN:
        import csv
        if isinstance(y_test, np.ndarray) or isinstance(y_test, pd.DataFrame
            ) or isinstance(y_test, pd.Series):
            shape_size = y_test.shape
        elif isinstance(y_test, list):
            shape_size = len(y_test)
        else:
            shape_size = 'any'
        check_type = type(y_test)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('y_test')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            if 'test_df' not in TANGSHAN:
                import csv
                if isinstance(test_df, np.ndarray) or isinstance(test_df,
                    pd.DataFrame) or isinstance(test_df, pd.Series):
                    shape_size = test_df.shape
                elif isinstance(test_df, list):
                    shape_size = len(test_df)
                else:
                    shape_size = 'any'
                check_type = type(test_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('test_df')
                    writer = csv.writer(f)
                    writer.writerow(['test_df', 25, check_type, shape_size])
            writer.writerow(['y_test', 47, check_type, shape_size])
    X_train = pd.concat([X_train, test_features_df])
    if 'test_features_df' not in TANGSHAN:
        import csv
        if isinstance(test_features_df, np.ndarray) or isinstance(
            test_features_df, pd.DataFrame) or isinstance(test_features_df,
            pd.Series):
            shape_size = test_features_df.shape
        elif isinstance(test_features_df, list):
            shape_size = len(test_features_df)
        else:
            shape_size = 'any'
        check_type = type(test_features_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('test_features_df')
            writer = csv.writer(f)
            writer.writerow(['test_features_df', 48, check_type, shape_size])
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['X_train', 48, check_type, shape_size])
    y_train = pd.concat([y_train, pd.DataFrame([-1] * len(test_features_df))])
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['y_train', 49, check_type, shape_size])
    params = {'n_neighbors': [3, 5, 7, 11], 'gamma': [1, 5, 10, 20, 50],
        'kernel': ['rbf', 'knn'], 'alpha': [0.05, 0.1, 0.2, 0.5],
        'max_iter': [10, 50, 100, 250]}
    if 'params' not in TANGSHAN:
        import csv
        if isinstance(params, np.ndarray) or isinstance(params, pd.DataFrame
            ) or isinstance(params, pd.Series):
            shape_size = params.shape
        elif isinstance(params, list):
            shape_size = len(params)
        else:
            shape_size = 'any'
        check_type = type(params)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('params')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['params', 55, check_type, shape_size])
    label_model = LabelSpreading(gamma=100, n_neighbors=15, kernel='knn',
        max_iter=10)
    label_model.fit(X_train, y_train)
    if 'label_model' not in TANGSHAN:
        import csv
        if isinstance(label_model, np.ndarray) or isinstance(label_model,
            pd.DataFrame) or isinstance(label_model, pd.Series):
            shape_size = label_model.shape
        elif isinstance(label_model, list):
            shape_size = len(label_model)
        else:
            shape_size = 'any'
        check_type = type(label_model)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('label_model')
            writer = csv.writer(f)
            writer.writerow(['label_model', 58, check_type, shape_size])
    predictions = label_model.predict(X_test)
    if 'predictions' not in TANGSHAN:
        import csv
        if isinstance(predictions, np.ndarray) or isinstance(predictions,
            pd.DataFrame) or isinstance(predictions, pd.Series):
            shape_size = predictions.shape
        elif isinstance(predictions, list):
            shape_size = len(predictions)
        else:
            shape_size = 'any'
        check_type = type(predictions)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('predictions')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['predictions', 60, check_type, shape_size])
    conf_matrix = confusion_matrix(y_test, predictions)
    if 'conf_matrix' not in TANGSHAN:
        import csv
        if isinstance(conf_matrix, np.ndarray) or isinstance(conf_matrix,
            pd.DataFrame) or isinstance(conf_matrix, pd.Series):
            shape_size = conf_matrix.shape
        elif isinstance(conf_matrix, list):
            shape_size = len(conf_matrix)
        else:
            shape_size = 'any'
        check_type = type(conf_matrix)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('conf_matrix')
            writer = csv.writer(f)
            writer.writerow(['conf_matrix', 61, check_type, shape_size])
    accuracy = sum([conf_matrix[i][i] for i in range(len(conf_matrix))]
        ) / np.sum(conf_matrix)
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['range', 63, check_type, shape_size])
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
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['accuracy', 63, check_type, shape_size])
    if 'sum' not in TANGSHAN:
        import csv
        if isinstance(sum, np.ndarray) or isinstance(sum, pd.DataFrame
            ) or isinstance(sum, pd.Series):
            shape_size = sum.shape
        elif isinstance(sum, list):
            shape_size = len(sum)
        else:
            shape_size = 'any'
        check_type = type(sum)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('sum')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            if 'test_df' not in TANGSHAN:
                import csv
                if isinstance(test_df, np.ndarray) or isinstance(test_df,
                    pd.DataFrame) or isinstance(test_df, pd.Series):
                    shape_size = test_df.shape
                elif isinstance(test_df, list):
                    shape_size = len(test_df)
                else:
                    shape_size = 'any'
                check_type = type(test_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('test_df')
                    writer = csv.writer(f)
                    writer.writerow(['test_df', 25, check_type, shape_size])
            writer.writerow(['sum', 63, check_type, shape_size])
    accuracies.append(accuracy)
if 'enumerate' not in TANGSHAN:
    import csv
    if isinstance(enumerate, np.ndarray) or isinstance(enumerate, pd.DataFrame
        ) or isinstance(enumerate, pd.Series):
        shape_size = enumerate.shape
    elif isinstance(enumerate, list):
        shape_size = len(enumerate)
    else:
        shape_size = 'any'
    check_type = type(enumerate)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('enumerate')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['enumerate', 43, check_type, shape_size])
if 'fold' not in TANGSHAN:
    import csv
    if isinstance(fold, np.ndarray) or isinstance(fold, pd.DataFrame
        ) or isinstance(fold, pd.Series):
        shape_size = fold.shape
    elif isinstance(fold, list):
        shape_size = len(fold)
    else:
        shape_size = 'any'
    check_type = type(fold)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('fold')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['fold', 43, check_type, shape_size])
print('Avg acc:', np.mean(accuracies))
if 'accuracies' not in TANGSHAN:
    import csv
    if isinstance(accuracies, np.ndarray) or isinstance(accuracies, pd.
        DataFrame) or isinstance(accuracies, pd.Series):
        shape_size = accuracies.shape
    elif isinstance(accuracies, list):
        shape_size = len(accuracies)
    else:
        shape_size = 'any'
    check_type = type(accuracies)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('accuracies')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['accuracies', 66, check_type, shape_size])
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['print', 66, check_type, shape_size])
label_model = LabelSpreading(gamma=50)
label_model.fit(all_features, all_labels)
prediction_vectors = []
for i, j in zip(test_ids, label_model.predict(test_features_df)):
    prediction_vectors.append([i, reverse_type_mapping[j]])
    if 'prediction_vectors' not in TANGSHAN:
        import csv
        if isinstance(prediction_vectors, np.ndarray) or isinstance(
            prediction_vectors, pd.DataFrame) or isinstance(prediction_vectors,
            pd.Series):
            shape_size = prediction_vectors.shape
        elif isinstance(prediction_vectors, list):
            shape_size = len(prediction_vectors)
        else:
            shape_size = 'any'
        check_type = type(prediction_vectors)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('prediction_vectors')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['prediction_vectors', 72, check_type, shape_size])
    if 'reverse_type_mapping' not in TANGSHAN:
        import csv
        if isinstance(reverse_type_mapping, np.ndarray) or isinstance(
            reverse_type_mapping, pd.DataFrame) or isinstance(
            reverse_type_mapping, pd.Series):
            shape_size = reverse_type_mapping.shape
        elif isinstance(reverse_type_mapping, list):
            shape_size = len(reverse_type_mapping)
        else:
            shape_size = 'any'
        check_type = type(reverse_type_mapping)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('reverse_type_mapping')
            if 'train_df' not in TANGSHAN:
                import csv
                if isinstance(train_df, np.ndarray) or isinstance(train_df,
                    pd.DataFrame) or isinstance(train_df, pd.Series):
                    shape_size = train_df.shape
                elif isinstance(train_df, list):
                    shape_size = len(train_df)
                else:
                    shape_size = 'any'
                check_type = type(train_df)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('train_df')
                    writer = csv.writer(f)
                    writer.writerow(['train_df', 24, check_type, shape_size])
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
                    writer.writerow(['i', 24, check_type, shape_size])
            if 'comb' not in TANGSHAN:
                import csv
                if isinstance(comb, np.ndarray) or isinstance(comb, pd.
                    DataFrame) or isinstance(comb, pd.Series):
                    shape_size = comb.shape
                elif isinstance(comb, list):
                    shape_size = len(comb)
                else:
                    shape_size = 'any'
                check_type = type(comb)
                with open('tas.csv', 'a+') as f:
                    TANGSHAN.append('comb')
                    writer = csv.writer(f)
                    writer.writerow(['comb', 24, check_type, shape_size])
            writer = csv.writer(f)
            writer.writerow(['reverse_type_mapping', 72, check_type,
                shape_size])
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['zip', 71, check_type, shape_size])
predictions_df = pd.DataFrame(prediction_vectors)
predictions_df.columns = ['id', 'type']
predictions_df.to_csv('submission_rbf.csv', index=False)
if 'predictions_df' not in TANGSHAN:
    import csv
    if isinstance(predictions_df, np.ndarray) or isinstance(predictions_df,
        pd.DataFrame) or isinstance(predictions_df, pd.Series):
        shape_size = predictions_df.shape
    elif isinstance(predictions_df, list):
        shape_size = len(predictions_df)
    else:
        shape_size = 'any'
    check_type = type(predictions_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('predictions_df')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['predictions_df', 75, check_type, shape_size])
label_model = LabelSpreading(n_neighbors=15, kernel='knn', max_iter=10)
label_model.fit(all_features, all_labels)
if 'all_features' not in TANGSHAN:
    import csv
    if isinstance(all_features, np.ndarray) or isinstance(all_features, pd.
        DataFrame) or isinstance(all_features, pd.Series):
        shape_size = all_features.shape
    elif isinstance(all_features, list):
        shape_size = len(all_features)
    else:
        shape_size = 'any'
    check_type = type(all_features)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('all_features')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['all_features', 78, check_type, shape_size])
prediction_vectors = []
for i, j in zip(test_ids, label_model.predict(test_features_df)):
    prediction_vectors.append([i, reverse_type_mapping[j]])
if 'j' not in TANGSHAN:
    import csv
    if isinstance(j, np.ndarray) or isinstance(j, pd.DataFrame) or isinstance(j
        , pd.Series):
        shape_size = j.shape
    elif isinstance(j, list):
        shape_size = len(j)
    else:
        shape_size = 'any'
    check_type = type(j)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('j')
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
                writer.writerow(['train_df', 24, check_type, shape_size])
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
                writer.writerow(['i', 24, check_type, shape_size])
        if 'comb' not in TANGSHAN:
            import csv
            if isinstance(comb, np.ndarray) or isinstance(comb, pd.DataFrame
                ) or isinstance(comb, pd.Series):
                shape_size = comb.shape
            elif isinstance(comb, list):
                shape_size = len(comb)
            else:
                shape_size = 'any'
            check_type = type(comb)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('comb')
                writer = csv.writer(f)
                writer.writerow(['comb', 24, check_type, shape_size])
        writer = csv.writer(f)
        writer.writerow(['j', 80, check_type, shape_size])
predictions_df = pd.DataFrame(prediction_vectors)
predictions_df.columns = ['id', 'type']
predictions_df.to_csv('submission_knn.csv', index=False)
