# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

homesite_df = pd.read_csv("../input/train.csv")
test_df     = pd.read_csv("../input/test.csv")

# preview the data
homesite_df.head()

homesite_df.info()
print("----------------------------")
test_df.info()

homesite_df = homesite_df.drop(['QuoteNumber'], axis=1)

homesite_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)

sns.countplot(x="QuoteConversion_Flag", data=homesite_df)

homesite_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

from sklearn import preprocessing

for f in homesite_df.columns:
    if homesite_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(homesite_df[f].values) + list(test_df[f].values)))
        homesite_df[f] = lbl.transform(list(homesite_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))
        
X_train = homesite_df.drop("QuoteConversion_Flag",axis=1)
Y_train = homesite_df["QuoteConversion_Flag"]
X_test  = test_df.drop("QuoteNumber",axis=1).copy()

params = {"objective": "binary:logistic"}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 30)
Y_pred = gbm.predict(X_test_xgb)

submission = pd.DataFrame()
submission["QuoteNumber"]          = test_df["QuoteNumber"]
submission["QuoteConversion_Flag"] = Y_pred

submission.to_csv('homesite.csv', index=False)