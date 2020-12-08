import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb
import lightgbm as lgb
from collections import Counter
from datetime import datetime
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Input data files are available in the "../input/" directory.
 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/kkbox-churn/user_label_201703.csv', dtype={'is_churn': 'int8'})

df_test = pd.read_csv('../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv')

user_log1 =  pd.read_csv('../input/kkbox-churn-prediction-challenge/user_logs.csv', nrows=36000000)
user_log =  pd.read_csv('../input/kkbox-churn-prediction-challenge/user_logs_v2.csv')
user_log = user_log.append(user_log1, ignore_index=True)
del user_log['date']
#group by msno
counts = user_log.groupby('msno')['total_secs'].count().reset_index()
counts.columns = ['msno', 'days_listened']
sums = user_log.groupby('msno').sum().reset_index()
user_log = sums.merge(counts, how='inner', on='msno')
print (str(np.shape(user_log)) + " -- New size of data matches unique member count")
#find avg seconds played per song
user_log['secs_per_song'] = user_log['total_secs'].div(user_log['num_25']+user_log['num_50']+user_log['num_75']+user_log['num_985']+user_log['num_100'])

training = df_train.merge(user_log, how='left', on='msno')
test_data = df_test.merge(user_log, how='left', on='msno')
del user_log
del user_log1
#==================== Finish cleaning user logs ===============================

df_transactions = pd.read_csv('../input/mytransaction/transactions_compiled.csv')
df_transactions = df_transactions.drop(['is_cancel','transaction_date','membership_expire_date'], axis=1)

#df_transactions['transaction_date']= pd.to_datetime(df_transactions['transaction_date'])
#df_transactions['membership_expire_date']= pd.to_datetime(df_transactions['membership_expire_date'])

training = training.merge(df_transactions, how='left', on='msno')
test_data = test_data.merge(df_transactions, how='left', on='msno')
del df_transactions

df_trans_price = pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions.csv')
df_trans_price2 = pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions_v2.csv')
df_trans_price = df_trans_price.append(df_trans_price2, ignore_index=True)

#remove col 'membership_expire_date','transaction_date','is_cancel','actual_amount_paid'
del df_trans_price['membership_expire_date']
del df_trans_price['transaction_date']
del df_trans_price['is_cancel']
#del df_trans_price['actual_amount_paid']
#del df_trans_price['payment_method_id']
df_trans_price = df_trans_price.drop_duplicates()
df_trans_price = df_trans_price.groupby('msno').mean().reset_index()

training = training.merge(df_trans_price, how='left', on='msno')
test_data = test_data.merge(df_trans_price, how='left', on='msno')
del df_trans_price
del df_trans_price2

#================ Finish cleaning transaction logs =============================

df_members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members_v3.csv')
#remove bd due to outliners
del df_members['bd']
del df_members['city']

# process of binning
#df_members['registered_via'].replace([-1, 1, 2, 5, 6, 8, 10, 11, 13, 14, 16, 17, 18, 19], 1, inplace = True)

#convert gender to int value
gender = {'male':1, 'female':2}
df_members['gender'] = df_members['gender'].map(gender)

#get number days from 31 mar 17 to reg init date
current = datetime.strptime('20170331', "%Y%m%d").date()
df_members['num_days'] = df_members.registration_init_time.apply(lambda x: (current - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN" )
del df_members['registration_init_time']

training = training.merge(df_members, how='left', on='msno')
test_data = test_data.merge(df_members, how='left', on='msno')
# ==================== Finish cleaning members data ===========================m

training = training.fillna(-1)
test_data=test_data.fillna(-1)

corrmat = training[training.columns[1:]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sb.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True);
plt.show()

#remove highly corelated inputs
#del training['payment_plan_days']
#del test_data['payment_plan_days']
#del training['payment_plan_days']
#del test_data['payment_plan_days']
#del user_data['payment_plan_days']

cols = [c for c in training.columns if c not in ['is_churn','msno']]
X = training[cols]
Y = training['is_churn']
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state = seed)

lgb_params = {
    'learning_rate': 0.01,
    'application': 'binary',
    'max_depth': 40,
    'num_leaves': 3300,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
d_train = lgb.Dataset(X_train, label=Y_train)
d_valid = lgb.Dataset(X_validation, label=Y_validation)
watchlist = [d_train, d_valid]

#cross validate score
#scoring = 'roc_auc'
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('DT', DecisionTreeClassifier()))

# evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)

model = lgb.train(lgb_params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 

#print('Plot feature importances...')
#ax = lgb.plot_importance(model, max_num_features=20)
#plt.show()

lgb_pred = model.predict(test_data[cols])

test_data['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
test_data[['msno','is_churn']].to_csv('lgb_result.csv', index=False)

