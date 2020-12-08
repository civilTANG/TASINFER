# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:38:00 2018

@author: JARD
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,15)


def data_prep(train, test):
    
    print("train has shape {0}, test has shape {1}".format(train.shape, test.shape))
    print("columns are : {0}".format(train.columns))
    
    print(train.dtypes)
    print(pd.isnull(train).sum())
    print(pd.isnull(test).sum())
    
    #### suppress too high premiums train["Premium"].hist(bins=30) ---> 8000 can also take percentiles as an improvement
    train = train.loc[train["Premium"]< 8000]
    
    #### power to weight ratio  ---> see there are observations with ratio = 0 , might be outliers, label them with 1
    train['possible_outliers'] = np.where(train["Power to Weight Ratio"]==0,1,0)
    test['possible_outliers'] = np.where(test["Power to Weight Ratio"]==0,1,0)
    
    #### make the same for all variables might be helpful
    # ....
    # .... #
    
    ### delete variables that are redundant or not meaningful
    variables_to_drop = ['sa1_11digit', 'sa1_7digit', 'Time of quote', 'Suburb', 'Country of Manufacture','NVIC', 'ID.1',
                         "Driver Age Detailed", 'Driver Age Band', 'SEIFA Score Banded', 'Vehicle Age Band', 'Vehicle Market Value Band',
                         'Vehicle Market Value Detailed', 'GNAF_PID', "State"]
    
    train = train.drop(variables_to_drop, axis=1) ### axis =0 ---> drop rows, axis=1 ----> drop column
    test  = test.drop(variables_to_drop, axis=1)
    
    ###### get rid of variables having one modality for the whole column
    for col in train.columns: 
       if len(train[col].unique()) <= 1:
           del train[col]
           del test[col]
           
    ###### encode variables having 2 modalities as a dummy : e.g Male/Female ----> 0/1      
    for col in train.columns: 
       if len(train[col].unique()) == 2:
           dico = {train[col].unique()[0]:0, train[col].unique()[1]:1}
           train[col] =train[col].map(dico)
           test[col] = test[col].map(dico)
            
    ###### for all columns not real (O means object) we dummify each modality, except for variables having more than 15 modalities
    ###### object variables with more than 15 modalities are deleted as a simple rule 
    cols_object = [x for x in train.columns if train[x].dtypes == "O"]
    for col in cols_object:
        if  col != "ID":
            if len(train[col].unique())< 15 :
                a = pd.get_dummies(train[col], prefix = col)
                train = pd.concat([a, train], axis = 1)
                del train[col]
                
                a = pd.get_dummies(test[col], prefix = col)
                test = pd.concat([a, test], axis = 1)
                del test[col]
                
            else:
                del train[col]
                del test[col]
    
    ##### make sure we have exactly same columns in train and test, otherwise the algorithm won't work
    ##### the difference of shape should come from the variable Premium in train but not in test set
    common_columns = list(set.intersection(set(test.columns), set(train.columns)))
    
    test=test[common_columns]
    train=train[common_columns + ["Premium"]]
    
    print("train has shape {0}, test has shape {1}".format(train.shape, test.shape))
     
    return train, test

  
def modelling_xgb(train, Y_label, test, params):
    
    data = train.reset_index(drop=True).copy()
    
    sample_submission = pd.DataFrame(test["ID"])
    sample_submission['Premium'] = 0
    
    X = data.drop([Y_label, "ID"],axis= 1)
    y = np.log(data[Y_label])
    
    ### declare the number of time we split train dataset into train and validation consecutive sets (5 times 80%/20%)
    k_fold = 5 
    Kf = KFold(n_splits=k_fold, shuffle= True, random_state= 36520)
    
    avg_mape = []
    avg_rmse = []
    avg_mae = []
    
    #### declare a dataset that will be filled by variable importance in the end
    dataset_importance = pd.DataFrame([], columns = ["variables", "importance"])
    dataset_importance["variables"] = X.columns
    
    #### for each fold we have a set of indexes to detect which rows are for training and which are for testing
    for i, (train_i, test_i) in enumerate(Kf.split(X, y)):
        
        X_train, X_test = X.loc[train_i], X.loc[test_i] ### create features for train set and features for validation set
        y_train, y_test = y.loc[train_i], y.loc[test_i] ### create target for train set and features for validation set
        
        eval_set  = [(X_train,y_train), (X_test,y_test)]
        
        ### declare the model we want to train, here this is a gradient boosting regressor
        clf = xgb.XGBRegressor(**params)
        clf.fit(X_train, y_train, ###### X_train = features , y_train = Premium (the target to predict)
                eval_set=eval_set,
                eval_metric= "mae", 
                early_stopping_rounds=30, #### the maximum number of iterations before stopping the training when the error starts to increase again on validation set
                verbose= 1)
        
        dataset_importance["importance"] = clf.feature_importances_
        
        preds = clf.predict(X_test) #### predict the value for the validataion set to calculate the error afterward
        pp = pd.DataFrame(np.transpose([np.exp(y_test).tolist(), np.exp(preds).tolist()]), columns = ["true", "pred"])
    
        print("[Fold {0}] MAPE : {1}, RMSE : {2}, MAE {3}".format(i ,(abs(pp["true"] - pp["pred"])*100/pp["true"]).mean(), np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ), abs(pp["true"] - pp["pred"]).mean()  ))
        avg_mape.append((abs(pp["true"] - pp["pred"])*100/pp["true"]).mean())
        avg_rmse.append(np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ))
        avg_mae.append(abs(pp["true"] - pp["pred"]).mean())
        
        sample_submission['Premium'] += np.exp(clf.predict(test.drop("ID",axis=1))) #### predict for the submission set : test
        
    sample_submission['Premium']  =  sample_submission['Premium']/float(k_fold)
    xgb.plot_importance(clf)
       
    print("_"*40)
    print("[OVERALL] MAPE : {0}, RMSE : {1}, MAE {2}".format(np.mean(avg_mape), np.mean(avg_rmse), np.mean(avg_mae)))
    
    return clf, pp, dataset_importance.sort_values("importance"), sample_submission


def results_analysis(preds):
    
    fig, ax = plt.subplots()

    ax.scatter(preds["true"], preds["pred"], alpha = 0.5, s= 5, label= ["true", "pred"])
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color= "red")
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('true', fontsize=16)
    plt.ylabel('prediction', fontsize=16)
    
    
#if __name__ == "__main__":
#    
#    #### change the path for the folder where you saved your train/test datasets
##    path = r"C:\Users\JARD\Documents\projects\kaggle\kernel1\data"
#    
#    train= pd.read_csv("../input/train.csv", header = 0)
#    test = pd.read_csv("../input/test.csv", header = 0)
#    
#    ### data preparation
#    train1, test1 = data_prep(train, test)
#    
#    ### modelling
#    params = {"objective" : 'reg:linear', #### objective is to do a regression
#              "n_estimators": 300,     #### number of trees to build, the higher the more the model is complex
#              "learning_rate": 0.15,   #### the weight on each prediction make by each tree, the smaller the less we overfit
#              "subsample": 0.75,       #### the proportion of data used to train one tree (this is randomly sampled for each tree)
#              "colsample_bytree":0.7,  #### the proportion of columns to use to build each tree (also randomly sampled)
#              "max_depth":4,           #### the maximum depth of a tree 
#              "gamma":0,               #### penalization to make predictions more conservative, the higher the less we overfit
#              "min_child_weight":2,    #### the minimum number of individuals to have in each final leaf of a tree
#              "seed" : 7666 }          #### the seed used to make uniform sampling (enables to make reproductible results under same conditions)
#     
#    model, test_reponse, variable_importance, sample_submission = modelling_xgb(train1, "Premium", test1, params)
#    
#    ##### analysis results 
#    results_analysis(test_reponse)
#    
#    #### submit 
#    sample_submission.to_csv("../output/xgb_solution1.csv",index= False)