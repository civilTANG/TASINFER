import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
if 'test' not in TANGSHAN:
    import csv
    if isinstance(test, np.ndarray) or isinstance(test, pd.DataFrame
        ) or isinstance(test, pd.Series):
        shape_size = test.shape
    elif isinstance(test, list):
        shape_size = len(test)
    else:
        shape_size = 'any'
    check_type = type(test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test')
        writer = csv.writer(f)
        writer.writerow(['test', 23, check_type, shape_size])
subm = pd.read_csv('../input/sample_submission.csv')
df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna('unknown')
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
        writer.writerow(['df', 35, check_type, shape_size])
nrow_train = train.shape[0]
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
X = vectorizer.fit_transform(df)
if 'vectorizer' not in TANGSHAN:
    import csv
    if isinstance(vectorizer, np.ndarray) or isinstance(vectorizer, pd.
        DataFrame) or isinstance(vectorizer, pd.Series):
        shape_size = vectorizer.shape
    elif isinstance(vectorizer, list):
        shape_size = len(vectorizer)
    else:
        shape_size = 'any'
    check_type = type(vectorizer)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('vectorizer')
        writer = csv.writer(f)
        writer.writerow(['vectorizer', 43, check_type, shape_size])
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
        writer.writerow(['X', 43, check_type, shape_size])
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))
if 'col' not in TANGSHAN:
    import csv
    if isinstance(col, np.ndarray) or isinstance(col, pd.DataFrame
        ) or isinstance(col, pd.Series):
        shape_size = col.shape
    elif isinstance(col, list):
        shape_size = len(col)
    else:
        shape_size = 'any'
    check_type = type(col)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('col')
        writer = csv.writer(f)
        writer.writerow(['col', 51, check_type, shape_size])
loss = []
if 'loss' not in TANGSHAN:
    import csv
    if isinstance(loss, np.ndarray) or isinstance(loss, pd.DataFrame
        ) or isinstance(loss, pd.Series):
        shape_size = loss.shape
    elif isinstance(loss, list):
        shape_size = len(loss)
    else:
        shape_size = 'any'
    check_type = type(loss)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('loss')
        writer = csv.writer(f)
        writer.writerow(['loss', 59, check_type, shape_size])
for i, j in enumerate(col):
    print('===Fit ' + j)
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
            writer.writerow(['print', 65, check_type, shape_size])
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    if 'nrow_train' not in TANGSHAN:
        import csv
        if isinstance(nrow_train, np.ndarray) or isinstance(nrow_train, pd.
            DataFrame) or isinstance(nrow_train, pd.Series):
            shape_size = nrow_train.shape
        elif isinstance(nrow_train, list):
            shape_size = len(nrow_train)
        else:
            shape_size = 'any'
        check_type = type(nrow_train)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('nrow_train')
            writer = csv.writer(f)
            writer.writerow(['nrow_train', 69, check_type, shape_size])
    preds[:, (i)] = model.predict_proba(X[nrow_train:])[:, (1)]
    pred_train = model.predict_proba(X[:nrow_train])[:, (1)]
    if 'model' not in TANGSHAN:
        import csv
        if isinstance(model, np.ndarray) or isinstance(model, pd.DataFrame
            ) or isinstance(model, pd.Series):
            shape_size = model.shape
        elif isinstance(model, list):
            shape_size = len(model)
        else:
            shape_size = 'any'
        check_type = type(model)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('model')
            writer = csv.writer(f)
            writer.writerow(['model', 75, check_type, shape_size])
    print('ROC AUC:', roc_auc_score(train[j], pred_train))
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
            writer.writerow(['j', 77, check_type, shape_size])
    loss.append(roc_auc_score(train[j], pred_train))
    if 'pred_train' not in TANGSHAN:
        import csv
        if isinstance(pred_train, np.ndarray) or isinstance(pred_train, pd.
            DataFrame) or isinstance(pred_train, pd.Series):
            shape_size = pred_train.shape
        elif isinstance(pred_train, list):
            shape_size = len(pred_train)
        else:
            shape_size = 'any'
        check_type = type(pred_train)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('pred_train')
            writer = csv.writer(f)
            writer.writerow(['pred_train', 79, check_type, shape_size])
    if 'train' not in TANGSHAN:
        import csv
        if isinstance(train, np.ndarray) or isinstance(train, pd.DataFrame
            ) or isinstance(train, pd.Series):
            shape_size = train.shape
        elif isinstance(train, list):
            shape_size = len(train)
        else:
            shape_size = 'any'
        check_type = type(train)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('train')
            writer = csv.writer(f)
            writer.writerow(['train', 79, check_type, shape_size])
if 'i' not in TANGSHAN:
    import csv
    if isinstance(i, np.ndarray) or isinstance(i, pd.DataFrame) or isinstance(i
        , pd.Series):
        shape_size = i.shape
    elif isinstance(i, list):
        shape_size = len(i)
    else:
        shape_size = 'any'
    check_type = type(i)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('i')
        writer = csv.writer(f)
        writer.writerow(['i', 63, check_type, shape_size])
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
        writer = csv.writer(f)
        writer.writerow(['enumerate', 63, check_type, shape_size])
print('mean column-wise ROC AUC:', np.mean(loss))
submid = pd.DataFrame({'id': subm['id']})
if 'submid' not in TANGSHAN:
    import csv
    if isinstance(submid, np.ndarray) or isinstance(submid, pd.DataFrame
        ) or isinstance(submid, pd.Series):
        shape_size = submid.shape
    elif isinstance(submid, list):
        shape_size = len(submid)
    else:
        shape_size = 'any'
    check_type = type(submid)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('submid')
        writer = csv.writer(f)
        writer.writerow(['submid', 89, check_type, shape_size])
if 'subm' not in TANGSHAN:
    import csv
    if isinstance(subm, np.ndarray) or isinstance(subm, pd.DataFrame
        ) or isinstance(subm, pd.Series):
        shape_size = subm.shape
    elif isinstance(subm, list):
        shape_size = len(subm)
    else:
        shape_size = 'any'
    check_type = type(subm)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('subm')
        writer = csv.writer(f)
        writer.writerow(['subm', 89, check_type, shape_size])
submission = pd.concat([submid, pd.DataFrame(preds, columns=col)], axis=1)
if 'submission' not in TANGSHAN:
    import csv
    if isinstance(submission, np.ndarray) or isinstance(submission, pd.
        DataFrame) or isinstance(submission, pd.Series):
        shape_size = submission.shape
    elif isinstance(submission, list):
        shape_size = len(submission)
    else:
        shape_size = 'any'
    check_type = type(submission)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('submission')
        writer = csv.writer(f)
        writer.writerow(['submission', 91, check_type, shape_size])
if 'preds' not in TANGSHAN:
    import csv
    if isinstance(preds, np.ndarray) or isinstance(preds, pd.DataFrame
        ) or isinstance(preds, pd.Series):
        shape_size = preds.shape
    elif isinstance(preds, list):
        shape_size = len(preds)
    else:
        shape_size = 'any'
    check_type = type(preds)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('preds')
        writer = csv.writer(f)
        writer.writerow(['preds', 91, check_type, shape_size])
submission.to_csv('submission.csv', index=False)
