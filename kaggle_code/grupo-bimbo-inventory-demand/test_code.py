import numpy as np
import pandas as pd
file_name = 'test'
df = pd.read_csv('../input/' + file_name + '.csv')
tabla_merete = df.shape[0]
print('A ' + file_name + ' halmazban szereplo rekordok szama: ' + str(
    tabla_merete))
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
            writer.writerow(['str', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('file_name')
        writer = csv.writer(f)
        writer.writerow(['file_name', 7, check_type, shape_size])
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
            writer.writerow(['str', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('print')
        writer = csv.writer(f)
        writer.writerow(['print', 7, check_type, shape_size])
if 'tabla_merete' not in TANGSHAN:
    import csv
    if isinstance(tabla_merete, np.ndarray) or isinstance(tabla_merete, pd.
        DataFrame) or isinstance(tabla_merete, pd.Series):
        shape_size = tabla_merete.shape
    elif isinstance(tabla_merete, list):
        shape_size = len(tabla_merete)
    else:
        shape_size = 'any'
    check_type = type(tabla_merete)
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
            writer.writerow(['str', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tabla_merete')
        writer = csv.writer(f)
        writer.writerow(['tabla_merete', 7, check_type, shape_size])
    if 'tmp_df' not in TANGSHAN:
        import csv
        if isinstance(tmp_df, np.ndarray) or isinstance(tmp_df, pd.DataFrame
            ) or isinstance(tmp_df, pd.Series):
            shape_size = tmp_df.shape
        elif isinstance(tmp_df, list):
            shape_size = len(tmp_df)
        else:
            shape_size = 'any'
        check_type = type(tmp_df)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('tmp_df')
            writer = csv.writer(f)
            writer.writerow(['tmp_df', 22, check_type, shape_size])
oszlopok_szama = df.shape[1]
for i in range(1, oszlopok_szama):
    tmp_df = pd.notnull(df.iloc[:, (i)])
    nem_nulla_ertekek_szama = tmp_df.shape[0]
    print('A ' + file_name + ' halmazban szereplo ' + tmp_df.name +
        """ nevu oszlopban:
null ertekek szama: """ + str(tabla_merete -
        tmp_df.shape[0]))
    tmp_df = df.iloc[:, (i)]
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
                writer.writerow(['str', 20, check_type, shape_size])
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('df')
            writer = csv.writer(f)
            writer.writerow(['df', 18, check_type, shape_size])
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
                writer.writerow(['str', 20, check_type, shape_size])
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('i')
            writer = csv.writer(f)
            writer.writerow(['i', 18, check_type, shape_size])
    if tmp_df.dtype == 'int64':
        print('maximum erteke: ' + str(tmp_df.max()))
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
                writer.writerow(['str', 20, check_type, shape_size])
        print('minimum erteke: ' + str(tmp_df.min()))
        print('atlag erteke: ' + str(tmp_df.mean()))
        if 'tmp_df' not in TANGSHAN:
            import csv
            if isinstance(tmp_df, np.ndarray) or isinstance(tmp_df, pd.
                DataFrame) or isinstance(tmp_df, pd.Series):
                shape_size = tmp_df.shape
            elif isinstance(tmp_df, list):
                shape_size = len(tmp_df)
            else:
                shape_size = 'any'
            check_type = type(tmp_df)
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('tmp_df')
                writer = csv.writer(f)
                writer.writerow(['tmp_df', 22, check_type, shape_size])
        print('median erteke: ' + str(tmp_df.median()))
        print('szoras erteke: ' + str(tmp_df.std()) + '\n')
    if tmp_df.dtype == 'object':
        print('ertekek szama: ' + str(nem_nulla_ertekek_szama))
        if 'nem_nulla_ertekek_szama' not in TANGSHAN:
            import csv
            if isinstance(nem_nulla_ertekek_szama, np.ndarray) or isinstance(
                nem_nulla_ertekek_szama, pd.DataFrame) or isinstance(
                nem_nulla_ertekek_szama, pd.Series):
                shape_size = nem_nulla_ertekek_szama.shape
            elif isinstance(nem_nulla_ertekek_szama, list):
                shape_size = len(nem_nulla_ertekek_szama)
            else:
                shape_size = 'any'
            check_type = type(nem_nulla_ertekek_szama)
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
                    writer.writerow(['str', 20, check_type, shape_size])
            with open('tas.csv', 'a+') as f:
                TANGSHAN.append('nem_nulla_ertekek_szama')
                writer = csv.writer(f)
                writer.writerow(['nem_nulla_ertekek_szama', 26, check_type,
                    shape_size])
        print('leggyakoribb ertek: ' + str(tmp_df.value_counts().idxmax()) +
            '\n')
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
            writer.writerow(['str', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('range')
        writer = csv.writer(f)
        writer.writerow(['range', 14, check_type, shape_size])
if 'oszlopok_szama' not in TANGSHAN:
    import csv
    if isinstance(oszlopok_szama, np.ndarray) or isinstance(oszlopok_szama,
        pd.DataFrame) or isinstance(oszlopok_szama, pd.Series):
        shape_size = oszlopok_szama.shape
    elif isinstance(oszlopok_szama, list):
        shape_size = len(oszlopok_szama)
    else:
        shape_size = 'any'
    check_type = type(oszlopok_szama)
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
            writer.writerow(['str', 20, check_type, shape_size])
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('oszlopok_szama')
        writer = csv.writer(f)
        writer.writerow(['oszlopok_szama', 14, check_type, shape_size])
print('')
