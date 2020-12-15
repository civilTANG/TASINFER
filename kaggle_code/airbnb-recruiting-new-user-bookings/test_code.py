import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subprocess import check_output
# print(check_output(['ls', r'D:\dataset\airbnb-recruiting-new-user-bookings']).decode('utf8'))

TANGSHAN = []

def remove_outliers(df, column, min_val, max_val):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >=
        max_val), np.NaN, col_values)
    return df


def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(' ', '_').replace('(', '').replace(')'
            , '').replace('/', '_').replace('-', '').lower()
        col_name = column_to_convert[:12] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[df[column_to_convert] == category, col_name] = 1
    return df


def convert_to_counts(df, id_col, column_to_convert):
    id_list = df[id_col].drop_duplicates()
    df_counts = df.loc[:, ([id_col, column_to_convert])]
    df_counts['count'] = 1
    df_counts = df_counts.groupby(by=[id_col, column_to_convert], as_index=
        False, sort=False).sum()
    new_df = df_counts.pivot(index=id_col, columns=column_to_convert,
        values='count')
    new_df = new_df.fillna(0)
    categories = list(df[column_to_convert].drop_duplicates())
    for category in categories:
        cat_name = str(category).replace(' ', '_').replace('(', '').replace(')'
            , '').replace('/', '_').replace('-', '').lower()
        col_name = column_to_convert + '_' + cat_name
        new_df.rename(columns={category: col_name}, inplace=True)
    return new_df


print('Reading in data...')
tr_filepath = r'D:\dataset\airbnb-recruiting-new-user-bookings/train_users_2.csv'
if 'tr_filepath' not in TANGSHAN:
    import csv
    if isinstance(tr_filepath, np.ndarray) or isinstance(tr_filepath, pd.
        DataFrame) or isinstance(tr_filepath, pd.Series):
        shape_size = tr_filepath.shape
    elif isinstance(tr_filepath, list):
        shape_size = len(tr_filepath)
    else:
        shape_size = 'any'
    check_type = type(tr_filepath)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tr_filepath')
        writer = csv.writer(f)
        writer.writerow(['tr_filepath', 91, check_type, shape_size])
df_train = pd.read_csv(tr_filepath, header=0, index_col=None)
te_filepath = r'D:\dataset\airbnb-recruiting-new-user-bookings/test_users.csv'
if 'te_filepath' not in TANGSHAN:
    import csv
    if isinstance(te_filepath, np.ndarray) or isinstance(te_filepath, pd.
        DataFrame) or isinstance(te_filepath, pd.Series):
        shape_size = te_filepath.shape
    else:
        if isinstance(te_filepath, list):
            shape_size = len(te_filepath)
        else:
            shape_size = 'any'
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
                writer = csv.writer(f)
                writer.writerow(['list', 11, check_type, shape_size])
    check_type = type(te_filepath)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('te_filepath')
        writer = csv.writer(f)
        writer.writerow(['te_filepath', 93, check_type, shape_size])
df_test = pd.read_csv(te_filepath, header=0, index_col=None)
if 'df_test' not in TANGSHAN:
    import csv
    if isinstance(df_test, np.ndarray) or isinstance(df_test, pd.DataFrame
        ) or isinstance(df_test, pd.Series):
        shape_size = df_test.shape
    else:
        if isinstance(df_test, list):
            shape_size = len(df_test)
        else:
            shape_size = 'any'
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
                writer = csv.writer(f)
                writer.writerow(['list', 11, check_type, shape_size])
    check_type = type(df_test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('df_test')
        writer = csv.writer(f)
        writer.writerow(['df_test', 94, check_type, shape_size])
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
if 'df_train' not in TANGSHAN:
    import csv
    if isinstance(df_train, np.ndarray) or isinstance(df_train, pd.DataFrame
        ) or isinstance(df_train, pd.Series):
        shape_size = df_train.shape
    elif isinstance(df_train, list):
        shape_size = len(df_train)
    else:
        shape_size = 'any'
    check_type = type(df_train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('df_train')
        writer = csv.writer(f)
        writer.writerow(['df_train', 100, check_type, shape_size])
print(df_all.axes)
features = list(df_all.axes[1])
for feature in features:
    print(feature, df_all[feature].nunique())
if 'features' not in TANGSHAN:
    import csv
    if isinstance(features, np.ndarray) or isinstance(features, pd.DataFrame
        ) or isinstance(features, pd.Series):
        shape_size = features.shape
    elif isinstance(features, list):
        shape_size = len(features)
    else:
        shape_size = 'any'
    check_type = type(features)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('features')
        writer = csv.writer(f)
        writer.writerow(['features', 107, check_type, shape_size])
for feature in features:
    if (feature != 'id' and feature != 'age' and feature !=
        'date_account_created' and feature != 'date_first_booking' and 
        feature != 'timestamp_first_active'):
        print(feature, df_all[feature].unique())
        if 'feature' not in TANGSHAN:
            import csv
            if isinstance(feature, np.ndarray) or isinstance(feature, pd.
                DataFrame) or isinstance(feature, pd.Series):
                shape_size = feature.shape
            else:
                if isinstance(feature, list):
                    shape_size = len(feature)
                else:
                    shape_size = 'any'
                if 'list' not in TANGSHAN:
                    import csv
                    if isinstance(list, np.ndarray) or isinstance(list, pd.
                        DataFrame) or isinstance(list, pd.Series):
                        shape_size = list.shape
                    elif isinstance(list, list):
                        shape_size = len(list)
                    else:
                        shape_size = 'any'
                    check_type = type(list)
                    with open('tas.csv', 'a+') as f:
                        TANGSHAN.append('list')
                        writer = csv.writer(f)
                        writer.writerow(['list', 11, check_type, shape_size])
            check_type = type(feature)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('feature')
                writer = csv.writer(f)
                writer.writerow(['feature', 111, check_type, shape_size])
print('Fixing timestamps...')
df_all['date_account_created'] = pd.to_datetime(df_all[
    'date_account_created'], format='%Y-%m-%d')
df_all['timestamp_first_active'] = pd.to_datetime(df_all[
    'timestamp_first_active'], format='%Y%m%d%H%M%S')
if 'df_all' not in TANGSHAN:
    import csv
    if isinstance(df_all, np.ndarray) or isinstance(df_all, pd.DataFrame
        ) or isinstance(df_all, pd.Series):
        shape_size = df_all.shape
    else:
        if isinstance(df_all, list):
            shape_size = len(df_all)
        else:
            shape_size = 'any'
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
                writer = csv.writer(f)
                writer.writerow(['list', 11, check_type, shape_size])
    check_type = type(df_all)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('df_all')
        writer = csv.writer(f)
        writer.writerow(['df_all', 119, check_type, shape_size])
df_all['date_account_created'].fillna(df_all.timestamp_first_active,
    inplace=True)
df_all.drop('date_first_booking', axis=1, inplace=True)
print('Fixing age column...')
if 'print' not in TANGSHAN:
    import csv
    if isinstance(print, np.ndarray) or isinstance(print, pd.DataFrame
        ) or isinstance(print, pd.Series):
        shape_size = print.shape
    else:
        if isinstance(print, list):
            shape_size = len(print)
        else:
            shape_size = 'any'
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
                writer = csv.writer(f)
                writer.writerow(['list', 11, check_type, shape_size])
    check_type = type(print)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('print')
        writer = csv.writer(f)
        writer.writerow(['print', 135, check_type, shape_size])
df_all = remove_outliers(df=df_all, column='age', min_val=15, max_val=90)
df_all['age'].fillna(-1, inplace=True)
print('Filling first_affiliate_tracked column...')
df_all['first_affiliate_tracked'].fillna(-1, inplace=True)
df_all['gender'].replace('-unknown-', np.nan, inplace=True)
categorical_features = ['affiliate_channel', 'affiliate_provider',
    'country_destination', 'first_affiliate_tracked', 'first_browser',
    'first_device_type', 'gender', 'language', 'signup_app', 'signup_method']
for categorical_feature in categorical_features:
    df_all[categorical_feature] = df_all[categorical_feature].astype('category'
        )
    if 'categorical_feature' not in TANGSHAN:
        import csv
        if isinstance(categorical_feature, np.ndarray) or isinstance(
            categorical_feature, pd.DataFrame) or isinstance(
            categorical_feature, pd.Series):
            shape_size = categorical_feature.shape
        elif isinstance(categorical_feature, list):
            shape_size = len(categorical_feature)
        else:
            shape_size = 'any'
        check_type = type(categorical_feature)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('categorical_feature')
            writer = csv.writer(f)
            writer.writerow(['categorical_feature', 162, check_type,
                shape_size])
if 'categorical_features' not in TANGSHAN:
    import csv
    if isinstance(categorical_features, np.ndarray) or isinstance(
        categorical_features, pd.DataFrame) or isinstance(categorical_features,
        pd.Series):
        shape_size = categorical_features.shape
    else:
        if isinstance(categorical_features, list):
            shape_size = len(categorical_features)
        else:
            shape_size = 'any'
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
                writer = csv.writer(f)
                writer.writerow(['list', 11, check_type, shape_size])
    check_type = type(categorical_features)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('categorical_features')
        writer = csv.writer(f)
        writer.writerow(['categorical_features', 161, check_type, shape_size])
df_all['gender'].value_counts(dropna=False).plot(kind='bar', color=
    '#FD5C64', rot=0)
plt.xlabel('Gender')
plt.savefig('gender_distribution.png')
