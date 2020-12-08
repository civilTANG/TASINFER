#inspirations:
# - https://www.kaggle.com/aharless/xgboost-cv-lb-284
# - https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
# - https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import datetime as dt
import gc

OPTIMIZE_ROUNDS = False

# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)

##########################################
# Main execution loop
##########################################
start_time = dt.datetime.now()
print("Started at ", start_time)

# Read data
train_df = pd.read_csv('../input/train.csv', na_values="-1") # .iloc[0:200,:]
test_df = pd.read_csv('../input/test.csv', na_values="-1")

print("Finished data loading at ", dt.datetime.now())

# from olivier
# GBM-specific correction of Oliver's list:
# the following features will not be in as they have too many NaN (>10%):
# - "ps_reg_03",
# -  ps_car_03_cat,
# -  ps_car_05_cat
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
]

# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),
    ('ps_reg_01', 'ps_car_04_cat'),
]

# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values
y = train_df['target']

start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    train_df[name1] = train_df[f1].apply(lambda x: str(x)) + "_" + train_df[f2].apply(lambda x: str(x))
    test_df[name1] = test_df[f1].apply(lambda x: str(x)) + "_" + test_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
    train_df[name1] = lbl.transform(list(train_df[name1].values))
    test_df[name1] = lbl.transform(list(test_df[name1].values))

    train_features.append(name1)

X = train_df[train_features]
test_df = test_df[train_features]

# impute NaN in train  and test dataframes - otherwise GBM won't work correctly
X = X.fillna(X.mean())
test_df = test_df.fillna(test_df.mean()) # TODO: better do it on the combined DF
print("Training set details:")
print(X.info())
print("Testing set details:")
print(test_df.info())

f_cats = [f for f in X.columns if "_cat" in f]

y_valid_pred = 0*y
y_test_pred = 0

print("Finished data pre-processing at ", dt.datetime.now())

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

# Set up classifier
model = GradientBoostingClassifier(
    learning_rate=0.05,
    min_samples_split=5000,
    min_samples_leaf=40,
    max_depth=7,
    max_features='sqrt',
    subsample=0.9,
    random_state=10,
    n_estimators=190)

print("Finished setting up CV folds and classifier at ", dt.datetime.now())
# Run CV

print("Started CV at ", dt.datetime.now())
for i, (train_index, test_index) in enumerate(kf.split(train_df)):

    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
    X_test = test_df.copy()
    print("\nFold ", i)

    # Enocode data
    for f in f_cats:
        X_train[f + "_avg"], X_valid[f + "_avg"], X_test[f + "_avg"] = target_encode(
            trn_series=X_train[f],
            val_series=X_valid[f],
            tst_series=X_test[f],
            target=y_train,
            min_samples_leaf=200,
            smoothing=10,
            noise_level=0
        )
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        eval_set = [(X_valid, y_valid)]
        fit_model = model.fit(X_train, y_train)
    else:
        fit_model = model.fit(X_train, y_train)

    # Generate validation predictions for this fold
    pred = model.predict_proba(X_valid)[:, 1]
    print("  Gini = ", eval_gini(y_valid, pred))
    y_valid_pred.iloc[test_index] = pred

    # Accumulate test set predictions
    y_test_pred += model.predict_proba(X_test)[:, 1]

    del X_test, X_train, X_valid, y_train

print("Finished CV at ", dt.datetime.now())

y_test_pred /= K  # Average test set predictions
print("Finished average test set predictions at ", dt.datetime.now())

# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('gbm_submit.csv', float_format='%.6f', index=False)

end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print ("Finished the advanced gbm script ... ")
print("Elapsed time: ", elapsed_time)