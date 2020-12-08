import gc
from abc import ABC, abstractmethod
import time
import ast
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
pd.set_option('max_columns', 50)

# Methodical sources:
# https://www.kaggle.com/jsaguiar/baseline-with-news
# https://wkirgsn.github.io/2018/02/10/auto-downsizing-dtypes/

class Company:
    """There's no business like mining business.
    Jewelry and gems for the middle class.
    Currently under investigation over legal affairs."""
    
    def __init__(self):
        self.miner = DataMiner()
        self.cleaner = MineralCleaner()
        self.director = Directrice()
        self.auditor = Auditor()
    
    def grow(self):
        """Any given company must pursue the only one thing 
        in order to survive."""

        self.miner.dig_in_the_mine(*self.auditor.open_the_mine())
        self.miner.ask_for_cleansing(self.cleaner)
        self.director.interrogate(self.miner)
        
        for yield_report in self.auditor.report_yield():
          ten_day_forecast_df = self.director.forecast(*yield_report)
          self.auditor.log_forecast(ten_day_forecast_df)
        
        self.auditor.file_away()


class CompanyEmployee(ABC):
    """Once you're in, you'll never get out.
    Everyone has a weak point."""
    def __init__(self):
        self.tra_df = None
    
    @staticmethod
    def clocking_work(work):
        """Work-life balance is a foreign word."""
        def do_work(*args, **kwargs):
            """Don't ask what the company can do for you 
            but instead what you can do for the company"""
            start_time = time.time()  # clock in
            ret = work(*args, **kwargs)
            end_time = time.time()  # clock out
            print('Clocking {:.3} seconds'.format(end_time-start_time))
            return ret
        return do_work


class Auditor(CompanyEmployee):
    """Checks the company environment for its integrity.
    Maintains a long-time friendship with the directrice.
    Plays golf with the CEO every wednesday."""
    
    def __init__(self):
        self.env = twosigmanews.make_env()
    
    def open_the_mine(self):
        """The mine is to be opened by authorized staff only.
        Recent incidents required tightening admission."""
        return self.env.get_training_data()
    
    def report_yield(self):
        """Confidential."""
        print('Start reporting yields..')
        return self.env.get_prediction_days()
        
    def log_forecast(self, prediction_df):
        """Corporate Compliance is a big thing.
        Every CEO needs a trustworthy internal audit, if you know what I mean."""
        self.env.predict(prediction_df)
    
    def file_away(self):
        """At the end of the day no one really knows what happend behind closed doors"""
        self.env.write_submission_file()


class MineralCleaner(CompanyEmployee):
    """Cleaning mineral specimens since 1858.
    There ain't no kind of stone he hasn't seen before.
    Spends lunchtime with the miner, his only confidant."""
    
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, conv_table=None):
        if conv_table is None:
            self.conversion_table = \
                {'int': [np.int8, np.int16, np.int32, np.int64],
                 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                 'float': [np.float32, ]}
        else:
            self.conversion_table = conv_table

    def _sound(self, k):
        """Press pressure sounding comes in handy for any gem collector."""
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    @CompanyEmployee.clocking_work
    def clean_minerals(self, df, verbose=False):
        """Oxalic and muriatic acid are his favorites."""
        print("Start cleaning minerals..")
        mem_usage_orig = df.memory_usage().sum() / self.memory_scale_factor
        
        ret_list = Parallel(n_jobs=1)(delayed(self._clean)
                                                (df[c], c, verbose) for c in
                                                df.columns)
        del df
        gc.collect()
        ret = pd.concat(ret_list, axis=1)
        
        mem_usage_new = ret.memory_usage().sum() / self.memory_scale_factor
        print(f"Reduced yield from {mem_usage_orig:.4f} MB to {mem_usage_new:.4f} MB.")
        return ret

    def _clean(self, s, colname, verbose):
        """When diluting always add acid to water, not water to acid."""
        # skip NaNs
        if s.isnull().any():
            if verbose:
                print(colname, 'has NaNs - Skip..')
            return s
        # detect kind of type
        coltype = s.dtype
        if np.issubdtype(coltype, np.integer):
            conv_key = 'int' if s.min() < 0 else 'uint'
        elif np.issubdtype(coltype, np.floating):
            conv_key = 'float'
        else:
            if verbose:
                print(colname, 'is', coltype, '- Skip..')
            return s
        # find right candidate
        for cand, cand_info in self._sound(conv_key):
            if s.max() <= cand_info.max and s.min() >= cand_info.min:

                if verbose:
                    print('convert', colname, 'to', str(cand))
                return s.astype(cand)


class DataMiner(CompanyEmployee):
    """Born in the mines, living in the dark, meant to perish in the ashes.
    Hacking is his purpose, dust his friend and the daily grind his fate."""
    
    merge_col_anchors = ['assetCode', 'date']
    
    @CompanyEmployee.clocking_work
    def dig_in_the_mine(self, market_df, news_df):
        """The first half of the day digging is to pay off the digging license"""
        print("Digging..")
        
        # chop trainset to speed up kernel
        start = datetime(2016, 6, 1, 0, 0, 0).date()
        market_df = market_df.loc[market_df['time'].dt.date >= start].reset_index(drop=True)
        news_df = news_df.loc[news_df['time'].dt.date >= start].reset_index(drop=True)
        
        self.tra_df = self.dig(market_df, news_df)
    
    def dig(self, market_df, news_df):
        """Rock for rock, hack for hack, day for day"""
        news_df = self.hack_for_news(news_df)
        market_df['date'] = market_df.time.dt.date
        return market_df.merge(news_df, how='left', on=self.merge_col_anchors)
    
    def hack_for_news(self, df):
        """The news gem is precious and the company offers 
        lucrative incentives for its nourishment"""
        
        drop_list = ['audiences', 'subjects', 'assetName', 
                    'headline', 'firstCreated', 'sourceTimestamp']
        df.drop(drop_list, axis=1, inplace=True)
        
        # Factorize categorical columns
        for col in ['headlineTag', 'provider', 'sourceId']:
            df[col], uniques = pd.factorize(df[col])
            del uniques
            
        # Unstack news_df across asset codes
        stacked_asset_codes = (df['assetCodes'].astype('object')
                                .apply(lambda x: list(ast.literal_eval(x)))
                                .apply(pd.Series)
                                .stack())
        stacked_asset_codes.index = stacked_asset_codes.index.droplevel(-1)
        stacked_asset_codes.name = 'assetCode'  # note: no trailing "s"
        df = df.join(stacked_asset_codes)
        df.drop(['assetCodes'], axis=1, inplace=True)
        
        # group by assetcode and date, aggregate some stats
        df['date'] = df.time.dt.date
        agg_agenda = ['mean', 'max']
        grouped = df.groupby(self.merge_col_anchors).agg(agg_agenda).astype('float32')
        grouped.columns = ['_'.join(c) for c in grouped.columns]
        
        return grouped.reset_index()
    
    def ask_for_cleansing(self, cleaner):
        """Every now and then some gems get dissolved accidentally.
        This is the official statement."""
        self.tra_df.drop(['date', 'universe', 'assetCode', 'assetName', 'time'], 
                        axis=1, inplace=True)
        self.tra_df = cleaner.clean_minerals(self.tra_df)
    

class Directrice(CompanyEmployee):
    """Call the director if things are getting serious. 
    She's keeping track of models and compiles forecasts for the company.
    Her career is the most important thing in her life and so she sticks at nothing. 
    Her sense of thievishness is infamous."""
    
    lgbm_params = {'boosting_type': 'gbdt',
        'colsample_bytree': 0.8,
        'learning_rate': 0.01,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'min_split_gain': .0,
        'n_estimators': 200,
        'n_jobs': -1,
        'num_leaves': 30,
        'random_state': 2018,
        'reg_alpha': .8,
        'reg_lambda': .4,
        'silent': True,
        'subsample': 1.0,
        'subsample_for_bin': int(2e5),
        'subsample_freq': 0
    }
    
    def __init__(self):
        """Subordinates are just means to an end."""
        super().__init__()
        self.miner = None
        self.models = [LGBMClassifier(**self.lgbm_params),]
    
    def interrogate(self, miner):
        """Shake a miner down."""
        self.tra_df = miner.tra_df.copy()
        # make the miner her personal tool
        self.miner = miner
        # Train models according to what the miner has unearthed
        self.train_models()
    
    @CompanyEmployee.clocking_work
    def train_models(self):
        bin_target = (self.tra_df.returnsOpenNextMktres10 >= 0).astype('int8')
        drop_cols = [c for c in ['returnsOpenNextMktres10', 'date', 'universe', 
                        'assetCode', 'assetName', 'time'] if c in self.tra_df.columns] 
        self.tra_df.drop(drop_cols, axis=1, inplace=True)
        
        print('Start training models..')
        for model in self.models:
            model.fit(self.tra_df, bin_target)
            
    def forecast(self, market_obs_df, news_obs_df, pred_templ_df):
        """Predict the future deposit in the mines that will be unearthed."""
        
        # make the miner put in unpaid extra work
        df = self.miner.dig(market_obs_df, news_obs_df)
        df = (df.loc[df.assetCode.isin(pred_templ_df.assetCode)]
                .drop(['assetName', 'time', 'date'], axis=1)
                .set_index('assetCode'))
        
        # heavy lifting
        for i, model in enumerate(self.models):
            df[f'pred_{i}'] = model.predict_proba(df)[:, 1]
        df['pred'] = df.loc[:, [c for c in df.columns if 'pred_' in c]].mean(axis=1)/ len(self.models)
        
        pred_templ_df = (pred_templ_df.set_index('assetCode')
                        .join(df.pred*2-1)
                        .drop(['confidenceValue'], axis=1)
                        .reset_index()
                        .rename(columns={'pred': 'confidenceValue'}))

        return pred_templ_df


if '__main__' == __name__:
    kaggliton = Company()
    kaggliton.grow()