import numpy as np
import pandas as pd
act_train_df = pd.read_csv('../input/act_train.csv', dtype={'people_id': np
    .str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
if 'act_train_df' not in TANGSHAN:
    import csv
    if isinstance(act_train_df, np.ndarray) or isinstance(act_train_df, pd.
        DataFrame) or isinstance(act_train_df, pd.Series):
        shape_size = act_train_df.shape
    elif isinstance(act_train_df, list):
        shape_size = len(act_train_df)
    else:
        shape_size = 'any'
    check_type = type(act_train_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('act_train_df')
        writer = csv.writer(f)
        writer.writerow(['act_train_df', 20, check_type, shape_size])
act_test_df = pd.read_csv('../input/act_test.csv', dtype={'people_id': np.
    str, 'activity_id': np.str}, parse_dates=['date'])
people_df = pd.read_csv('../input/people.csv', dtype={'people_id': np.str,
    'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])


def intersect(l_1, l_2):
    return list(set(l_1) & set(l_2))


def get_features(train, test):
    intersecting_features = intersect(train.columns, test.columns)
    intersecting_features.remove('people_id')
    intersecting_features.remove('activity_id')
    return sorted(intersecting_features)


def process_date(input_df):
    df = input_df.copy()
    return df.assign(year=lambda df: df.date.dt.year, month=lambda df: df.
        date.dt.month, day=lambda df: df.date.dt.day).drop('date', axis=1)


def process_activity_category(input_df):
    df = input_df.copy()
    return df.assign(activity_category=lambda df: df.activity_category.str.
        lstrip('type ').astype(np.int32))


def process_activities_char(input_df, columns_range):
    """
    Extract the integer value from the different char_* columns in the
    activities dataframes. Fill the missing values with -999 as well
    """
    df = input_df.copy()
    char_columns = [('char_' + str(i)) for i in columns_range]
    return df[char_columns].fillna('type -999').apply(lambda col: col.str.
        lstrip('type ').astype(np.int32)).join(df.drop(char_columns, axis=1))


def activities_processing(input_df):
    """
    This function combines the date, activity_category and char_*
    columns transformations.
    """
    df = input_df.copy()
    return df.pipe(process_date).pipe(process_activity_category).pipe(
        process_activities_char, range(1, 11))


def process_group_1(input_df):
    df = input_df.copy()
    return df.assign(group_1=lambda df: df.group_1.str.lstrip('group ').
        astype(np.int32))


def process_people_cat_char(input_df, columns_range):
    """
    Extract the integer value from the different categorical char_*
    columns in the people dataframe.
    """
    df = input_df.copy()
    cat_char_columns = [('char_' + str(i)) for i in columns_range]
    return df[cat_char_columns].apply(lambda col: col.str.lstrip('type ').
        astype(np.int32)).join(df.drop(cat_char_columns, axis=1))


def process_people_bool_char(input_df, columns_range):
    """
    Extract the integer value from the different boolean char_* columns in the
    people dataframe.
    """
    df = input_df.copy()
    boolean_char_columns = [('char_' + str(i)) for i in columns_range]
    return df[boolean_char_columns].apply(lambda col: col.astype(np.int32)
        ).join(df.drop(boolean_char_columns, axis=1))


def people_processing(input_df):
    """
    This function combines the date, group_1 and char_*
    columns (inclunding boolean and categorical ones) transformations.
    """
    df = input_df.copy()
    return df.pipe(process_date).pipe(process_group_1).pipe(
        process_people_cat_char, range(1, 10)).pipe(process_people_bool_char,
        range(10, 38))


def merge_with_people(input_df, people_df):
    """
    Merge (left) the given input dataframe with the people dataframe and
    fill the missing values with -999.
    """
    df = input_df.copy()
    return df.merge(people_df, how='left', on='people_id', left_index=True,
        suffixes=('_activities', '_people')).fillna(-999)
    if 'people_df' not in TANGSHAN:
        import csv
        if isinstance(people_df, np.ndarray) or isinstance(people_df, pd.
            DataFrame) or isinstance(people_df, pd.Series):
            shape_size = people_df.shape
        elif isinstance(people_df, list):
            shape_size = len(people_df)
        else:
            shape_size = 'any'
        check_type = type(people_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('people_df')
            writer = csv.writer(f)
            writer.writerow(['people_df', 151, check_type, shape_size])


processed_people_df = people_df.pipe(people_processing)
train_df = act_train_df.pipe(activities_processing).pipe(merge_with_people, 
if 'processed_people_df' not in TANGSHAN:
    import csv
    if isinstance(processed_people_df, np.ndarray) or isinstance(
        processed_people_df, pd.DataFrame) or isinstance(processed_people_df,
        pd.Series):
        shape_size = processed_people_df.shape
    elif isinstance(processed_people_df, list):
        shape_size = len(processed_people_df)
    else:
        shape_size = 'any'
    check_type = type(processed_people_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('processed_people_df')
        writer = csv.writer(f)
        writer.writerow(['processed_people_df', 161, check_type, shape_size]
            ), processed_people_df)
test_df = act_test_df.pipe(activities_processing).pipe(merge_with_people,
    processed_people_df)
if 'act_test_df' not in TANGSHAN:
    import csv
    if isinstance(act_test_df, np.ndarray) or isinstance(act_test_df, pd.
        DataFrame) or isinstance(act_test_df, pd.Series):
        shape_size = act_test_df.shape
    elif isinstance(act_test_df, list):
        shape_size = len(act_test_df)
    else:
        shape_size = 'any'
    check_type = type(act_test_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('act_test_df')
        writer = csv.writer(f)
        writer.writerow(['act_test_df', 162, check_type, shape_size])
features_list = get_features(train_df, test_df)
print('The merged features are: ')
print('\n'.join(features_list), '\n')
if 'features_list' not in TANGSHAN:
    import csv
    if isinstance(features_list, np.ndarray) or isinstance(features_list,
        pd.DataFrame) or isinstance(features_list, pd.Series):
        shape_size = features_list.shape
    elif isinstance(features_list, list):
        shape_size = len(features_list)
    else:
        shape_size = 'any'
    check_type = type(features_list)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('features_list')
        writer = csv.writer(f)
        writer.writerow(['features_list', 172, check_type, shape_size])
print('The train dataframe head is', '\n')
print(train_df.head())
if 'train_df' not in TANGSHAN:
    import csv
    if isinstance(train_df, np.ndarray) or isinstance(train_df, pd.DataFrame
        ) or isinstance(train_df, pd.Series):
        shape_size = train_df.shape
    elif isinstance(train_df, list):
        shape_size = len(train_df)
    else:
        shape_size = 'any'
    check_type = type(train_df)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train_df')
        writer = csv.writer(f)
        writer.writerow(['train_df', 174, check_type, shape_size])
print('The test dataframe head is', '\n')
print(test_df.head())
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
        writer.writerow(['test_df', 176, check_type, shape_size])
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
        writer.writerow(['print', 176, check_type, shape_size])
