# Predictor model: Random Forest, XGBoost, CatBoost, SVM, Logistic Regression, KNN ...?

from typing import Union
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.exceptions import FitFailedWarning

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import json
from pathlib import Path
from src.common.logger import init_logger
import os
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
import warnings

import ray
from ray.tune.sklearn import TuneSearchCV

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class Benchmark:
    model_tables = {'rf': RandomForestClassifier, 
                    'lr': LogisticRegression,
                    'svm': SVC,
                    'knn': KNeighborsClassifier,
                    'ada': AdaBoostClassifier,
                    'gb': GradientBoostingClassifier,
                    'et': ExtraTreesClassifier,
                    'xgb': XGBClassifier,
                    'cb': CatBoostClassifier
                    }
    
    BASEDIR = str(Path(__file__).parent)
    
    def __init__(self, 
                 imputed,
                 original,
                 logging_nm='benchmark', 
                 label_nm='fradulent',
                 scoring={'precision': 'precision', 'recall': 'recall', 'f1': 'f1', "AUC": "roc_auc"},
                 cat_cols=['location', 'employment_type', 'required_experience', 'required_education', 
                                'industry', 'function'],
                 cv=10,
                 seed=42,
                 save_cv_result=True
                 ) -> None:
        self.imputed = imputed
        self.original = original
        self.label_nm = label_nm
        self.cv = cv
        self.base_path = f'{self.BASEDIR}/{logging_nm}'
        self.result_path = f'{self.base_path}/results.json'
        self.scoring = scoring
        self._score_of_interest = 'AUC'
        self.logger = init_logger({'level': 'INFO', 'name': logging_nm})
        self.seed = seed
        self.save_cv_result = save_cv_result
        self.best_params = {}
        self.cat_cols = cat_cols
        # if current os is windows, set n_jobs to 1 else -1
        self.n_jobs = 1 if os.name == 'nt' else -1
        self.use_gpu = self._test_xgb_finds_gpu()
        
        ray.init(
                num_cpus=8,
                num_gpus=1 if self.use_gpu else 0
        )
    
    def preprocess(self, imputed_df, original_df, make_dummies=True, drop_text=True):
        df = imputed_df.copy(deep=True)

        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        
        if drop_text:
            df = df.drop(text_cols, axis=1)
        
        if make_dummies:
            df = pd.get_dummies(df, columns=self.cat_cols)
            
        # concat fradulent col to imputed_df from original_df
        df['fraudulent'] = original_df['fraudulent']
        
        return df


    def _run_cv(self, model, **kwargs):
        y = self.imputed[self.label_nm]
        X = self.imputed.drop(self.label_nm, axis=1)
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        
        try:
            cv_scores = cross_validate(model(**kwargs), X, y, cv=cv, scoring=self.scoring, )
            
        except FitFailedWarning as e:
            self.logger.error(f'FitFailedWarning: {e}')
            raise Exception(f'FitFailedWarning: {e}')
        
        return cv_scores
    
    def save_json(self, imputed, path):
        _path = Path(path)
        
        if not _path.parent.exists():
            _path.parent.mkdir(parents=True)
            
        with open(_path, 'w') as f:
            json.dump(imputed, f, indent=4)
            
    def load_json(self, path):
        _path = Path(path)
        
        if not _path.exists():
            raise FileNotFoundError(f'{path} does not exist.')
            
        with open(_path, 'r') as f:
            imputed = json.load(f)
            
        return imputed
    
    def _test_xgb_finds_gpu(capsys):
        """Check if XGBoost finds the GPU."""
        import numpy as np
        X = np.random.rand(2, 4)
        y = np.random.randint(0, 1, 2)

        try:
            xgb_model = XGBClassifier(
                tree_method="gpu_hist"
            )
            xgb_model.fit(X, y)
            return True
        except:
            return False
    
    def _search_best_params(self, param_range, skip_param_search, **kwargs):
        model_param_range = self._get_param_ranges(param_range, skip_param_search)
        BEST_PARMAS_PATH = f'{self.base_path}/{self._score_of_interest}/best_params.json'
        
        try:
            self.best_params = self.load_json(BEST_PARMAS_PATH)
            
        except FileNotFoundError:
            self.logger.info(f'{BEST_PARMAS_PATH} does not exist. Start searching best params...')
            self.best_params = {}
            
        search_n_jobs = 8 if self.n_jobs == -1 else self.n_jobs
        
        for nm, param_range in model_param_range.items():
            
            _drop_text = True
            _make_dummies = True
            
            # check if random state attribute exists in the model class
            if 'random_state' in self.model_tables[nm]().get_params().keys():
                model = self.model_tables[nm](random_state=self.seed)
                
            else:
                model = self.model_tables[nm]()
            
            if nm in self.best_params.keys():
                self.logger.info(f'{nm} already has best params. Skip searching best params.')
                continue
            
            if nm == 'xgb':
                if self.use_gpu:
                    param_range['tree_method'] = ['gpu_hist']
                    param_range['gpu_id'] = [0]
                    
                param_range['scale_pos_weight'] = [len(self.original[self.original[self.label_nm] == 0]) / self.original.shape[0]] # scale_pos_weight: negative ratio of label
                search_n_jobs = 2
                
            if nm == 'cb':
                param_range['max_depth'] = (3, 16)
                param_range['od_wait'] = [10]                
                search_n_jobs = 1   # intentionally set to 1 because catboost runs really slow if parallel search is executed from the ray
                model.set_params(cat_features=self.cat_cols)
                
                # _drop_text = True  # if you want to run the model also with text when using catboost, set drop_text to False
                _make_dummies = False   # catboost does not need dummies
            
            self.logger.info(f'Searching best params for {nm} with {search_n_jobs} jobs...')
            
            search = TuneSearchCV(
                model,
                param_range,
                search_optimization="bayesian",
                n_trials=20,
                early_stopping=False,
                max_iters=1,
                n_jobs=search_n_jobs,
                scoring=self.scoring,
                refit=self._score_of_interest,                
                cv=self.cv,
                use_gpu=self.use_gpu                
            )
            

            data = self.preprocess(self.imputed, self.original, drop_text=_drop_text, make_dummies=_make_dummies)
            X, y = data.drop(self.label_nm, axis=1), data[self.label_nm]
            
            search.fit(X, y)
            
            if self.save_cv_result:
                cv_result = search.cv_results_
                
                if not Path(f'{self.base_path}/cv_result/{self._score_of_interest}').exists():
                    Path(f'{self.base_path}/cv_result/{self._score_of_interest}').mkdir(parents=True)
                    
                pd.DataFrame.from_dict(cv_result).to_csv(f'{self.base_path}/cv_result/{self._score_of_interest}/{nm}.csv', index=False)
            
            self.best_params[nm] = {'params': search.best_params_, 'score': search.best_score_}
            
            # save the best params
            self.save_json(self.best_params, BEST_PARMAS_PATH)
            self.logger.info(f'Best params for {nm} on {self._score_of_interest} with {search.best_score_}: {search.best_params_}')
            self.logger.info(f'Finished searching best params for {nm}.')

    def _get_param_ranges(self, param_range_file, skip_param_search=False):
        if not (isinstance(param_range_file, dict) or isinstance(param_range_file, str)):
            raise TypeError('param_range_file must be a dict or a string.')
        
        if isinstance(param_range_file, str):
            try:
                model_param_range = self.load_json(param_range_file)
                
            except FileNotFoundError:
                self.logger.error(f'predefined parameter ranges on {param_range_file} does not exist.')
                raise FileNotFoundError(f'predefined parameter ranges on {param_range_file} does not exist.')
            
        elif isinstance(param_range_file, dict):
            model_param_range = param_range_file
            
        else:
            raise TypeError('param_range_file must be a dict or a string.')
        
        # check if the values in the models are list. If parameter search is skipped, this is not necessary
        if not skip_param_search:
            for k, parameters in model_param_range.items():
                for param_nm, v in parameters.items():
                    if not isinstance(v, list):
                        
                        if isinstance(v, (int, float, str)):
                            model_param_range[k][param_nm] = [v]
                        
                        else:
                            raise TypeError(f'Parameter ranges for {k} must be a list.')
                
        else:
            # if parameter search is skipped, the value of the model_param_range should not be any iterables or containers.
            for k, v in model_param_range.items():
                if isinstance(v, (list, tuple, dict, set)):
                    raise TypeError(f'Parameter ranges for {k} must not be an iterable when you want to skip the param search.'
                                    f'Please provide a single value. Given type: {type(v)}')
                
        return model_param_range
    
    def run(self, param_range: Union[dict, str], skip_param_search=False):
        # models: {model_nm: model_params, ...}
        
        import warnings
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        
        try:
            results = self.load_json(self.result_path)
            self.logger.info('Previous results found. Running benchmark on the loaded result set...')
            
        except FileNotFoundError:
            self.logger.info('No previous results found. Running benchmark on a new result set...')
            results = {}
        
        self._search_best_params(param_range, skip_param_search)
        
        for nm, params in self.best_params.items():
            try:
                if nm in self.model_tables.keys() and nm not in results.keys():
                    # only run if the model is defined and not in the resultss
                    self.logger.info(f'Running benchmark on {nm} with params {params}')
                    model = self.model_tables[nm]
                    cv_scores = self._run_cv(model, **params)
                    avg_scores = {k: v.mean() for k, v in cv_scores.items()}
                    results[nm] = {"cv_avg_scores": avg_scores, "params": params}
                    self.logger.info(f'Finished running benchmark on {nm}')
                    
                else:
                    results[nm] = 'Undefined'
                    
            except Exception as e:
                results[nm] = str(e)
                self.logger.error(f'Error running benchmark on {nm}: {e}')
                
            # save intermediate results
            self.save_json(results, self.result_path)
        
        ray.shutdown()
        return results
    

if __name__ == '__main__':
    # create synthethic debug dataset for Benchmark
    import pandas as pd
    import numpy as np
    
    imputed = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))
    imputed['fradulent'] = np.random.randint(0, 2, size=1000)
    
    # load the predefined parameter ranges
    param_range = "./imputed/param_range.json"
    
    bm = Benchmark(imputed, 
                   label_nm='fradulent', cv=5, 
                   logging_nm='debug')
    
    bm.run(param_range, skip_param_search=False)
    
    