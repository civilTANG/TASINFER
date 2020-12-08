import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from scipy import sparse

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

class SiteCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, left=0, right=10):
        self.left = left
        self.right = right
        self.columns_ = 0
    
    @staticmethod
    def _convert_to_counts(values):
        indptr = np.empty(values.shape[0]+1, dtype=int)
        indptr[0] = 0
        indices = np.empty(values.shape[0], dtype=object)
        data = np.empty(values.shape[0], dtype=object)
        offset = 0
    
        for i, session in enumerate(values):
            jj, counts = np.unique(session, return_counts=True)
            offset += len(jj)

            indptr[i+1] = offset
            data[i] = counts
            indices[i] = jj

        return csr_matrix((np.concatenate(data), np.concatenate(indices), indptr))[:, 1:]
        
    
    def fit(self, X, y=None):
        self.columns_ = X[:, self.left:self.right].astype(int).max()
        return self
    
    def transform(self, X, y='deprecated'):
        if self.columns_ == 0:
            raise NotFittedError
        
        M = SiteCountTransformer._convert_to_counts(X[:, self.left:self.right].astype(int)) / (self.right-self.left)
        if M.shape[1] > self.columns_:
            return M[:, :self.columns_]
        elif M.shape[1] < self.columns_:
            return sparse.hstack((M, np.zeros((M.shape[0], self.columns_ - M.shape[1]), dtype=M.dtype)), format='csr')
        else:
            return M

def write_to_submission_file(predicted_labels, out_file, target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

# Any results you write to the current directory are saved as output.
def write_submission(probas):
    write_to_submission_file(probas, 'submission.csv')

PATH_TO_DATA = '../input' # Input data files are available in the "../input/" directory.

kaggle_train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
kaggle_test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')

#reordering columns to simplyfy
kaggle_train_df = kaggle_train_df[['site%d' % i for i in range(1, 11)] + ['time%d' % i for i in range(1, 11)] + ['target']]
kaggle_test_df = kaggle_test_df[['site%d' % i for i in range(1, 11)] + ['time%d' % i for i in range(1, 11)]]

X_kaggle_train = np.concatenate((
    kaggle_train_df.iloc[:, :10].fillna(0).values,
    kaggle_train_df.iloc[:, 10:20].values
    ), axis=1)
    
y_kaggle_train = kaggle_train_df['target'].astype(int).values

X_kaggle_test = np.concatenate((
    kaggle_test_df.iloc[:, :10].fillna(0).values,
    kaggle_test_df.iloc[:, 10:20].values
    ), axis=1)
    
train_share = int(.7 * X_kaggle_train.shape[0])
X_train, y_train = X_kaggle_train[:train_share, :], y_kaggle_train[:train_share]
X_valid, y_valid  = X_kaggle_train[train_share:, :], y_kaggle_train[train_share:]

pipeline = Pipeline([
        ('site_counts', SiteCountTransformer()),
        ('estimator', LogisticRegressionCV(
            Cs=np.linspace(0.1, 10, 10), scoring='roc_auc', cv=TimeSeriesSplit(n_splits=5), random_state=17))
    ])

pipeline.fit(X_train, y_train)
print('ROC AUC on validation set: {}'.format(roc_auc_score(y_valid, pipeline.predict_proba(X_valid)[:, 1])))

pipeline.fit(X_kaggle_train, y_kaggle_train)
write_submission(pipeline.predict_proba(X_kaggle_test)[:, 1])