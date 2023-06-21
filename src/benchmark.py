# Predictor model: Random Forest, XGBoost, CatBoost, SVM, Logistic Regression, KNN ...?

from typing import Union
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import auc, average_precision_score, f1_score, recall_score
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
from src.sampler import Sampler
from src.feature_select import FeatureSelector

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import gc

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
    
    BASEDIR = str(Path(__file__).parent.parent)
    
    def __init__(self, 
                 original_df,
                 logging_nm='benchmark', 
                 label_nm='fradulent',
                 scoring={'precision': 'precision', 'recall': 'recall', 'f1': 'f1', "AUC": "roc_auc"},
                 text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits'],
                 cat_cols=['location', 'employment_type', 'required_experience', 'required_education', 
                                'industry', 'function'],
                 cv=10,
                 seed=42,
                 save_cv_result=True,   # save param search result to csv file
                 
                 sample_method=None,
                 sampler_kwargs={},
                 
                 feat_select_method=None,
                 feat_select_kwargs={},
                 
                 num_feat_to_sel : Union[int, float] = 0.75,
                 
                 skip_model = []
                 ) -> None:
        self.original_df = original_df
        self.label_nm = label_nm
        self.cv = cv
        
        self._score_of_interest = 'AUC'
        self.base_path = f'{self.BASEDIR}/benchmark_results/{logging_nm}'
        self.result_path = f'{self.base_path}/results_{self._score_of_interest}.json'
        self.scoring = scoring

        self.logger = init_logger({'level': 'INFO', 'name': logging_nm})
        self.seed = seed
        self.save_cv_result = save_cv_result
        self.skip_model = skip_model if isinstance(skip_model, list) else [skip_model]
        self.best_params = {}
        
        self.cat_cols = cat_cols
        
        if "department" in original_df.columns:
            self.cat_cols.append("department")
            
        self.text_cols = text_cols

        self.n_jobs = 4 if os.name == 'nt' else 6
        self.use_gpu = self._test_xgb_finds_gpu()   # must be called before ray.init()
        
        self.sampler = Sampler(method_nm=sample_method, **sampler_kwargs)
        
        self.feat_selector = FeatureSelector(method_nm=feat_select_method, **feat_select_kwargs)
        self.num_feat_to_sel = num_feat_to_sel
        
        if not ray.is_initialized():
            ray.init(
                    num_cpus=os.cpu_count()-3,  # if resouce is not enough, reduce this number
                    num_gpus=1 if self.use_gpu else 0
            )
    
    def preprocess(self, original_df, make_dummies=True, drop_text=True, sample=True, select_feature=True, selected_features=None):        
        df = original_df.copy(deep=True)
                        
        for cat_col in self.cat_cols:
            df[cat_col] = df[cat_col].fillna('Unspecified')
                        
        replace_edu_dict = {'Some High School Coursework': 'High School or equivalent',
                            'Vocational - HS Diploma': 'High School or equivalent',
                            'Vocational - Degree': 'High School or equivalent',
                            'Vocational': 'High School or equivalent',
                            'Some College Coursework Completed': "Bachelor's Degree",
                            'Doctorate': "Professional",
                            'nan': 'Unspecified'
        }
        
        df['required_education'] = df.required_education.replace(replace_edu_dict)
        
        # only use the first part of the location
        df['location'] = df['location'].apply(lambda x : str(x).split(',')[0])
        
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        
        if drop_text:
            df = df.drop(self.text_cols, axis=1)
            
        else:
            for text_col in self.text_cols:
                df[text_col] = df[text_col].fillna('Unspecified')
        
        if make_dummies:
            df = pd.get_dummies(df, columns=self.cat_cols)
            
        if selected_features is not None:
            df = df[selected_features + [self.label_nm]]
            select_feature = False
                
        X, X_test, y, y_test = train_test_split(df.drop(self.label_nm, axis=1), df[self.label_nm],
                                                            random_state=self.seed, stratify=df[self.label_nm])
        
        if select_feature: 
            k = int(X.shape[1] * self.num_feat_to_sel) if isinstance(self.num_feat_to_sel, float) else self.num_feat_to_sel
            
            if k > X.shape[1]:
                self.logger.warning(f'k is larger than the number of features. k is set to half of the features {X.shape[1]//2}')
                k = X.shape[1]//2
            
            self.logger.info(f"Running a feature selector: {self.feat_selector.get_method_nm()}")
        
            try:
                X_filtered = self.feat_selector.run(X, y, k)
                
            except ValueError as e:
                self.logger.warning(f"Feature selection failed. {e}")
                X_filtered = X
                
            self.logger.info(f"Running a feature selector finished.")
            
        else:
            X_filtered = X
            

        
        if sample is not None and sample is True:
            self.logger.info(f"Running a sampler: {self.sampler.get_method_nm()}")
            try:
                X_sampled, y_sampled = self.sampler.run(X_filtered, y)
            
            except ValueError as e:
                self.logger.warning(f"Sampling failed. {e}")
            
            self.logger.info(f"Running a sampler finished.")
            
        else:
            X_sampled, y_sampled = X_filtered, y
            
        return X_sampled, y_sampled, X_test, y_test
    
    def save_json(self, original_df, path):
        _path = Path(path)
        
        if not _path.parent.exists():
            _path.parent.mkdir(parents=True)
            
        with open(_path, 'w') as f:
            json.dump(original_df, f, indent=4)
            
    def load_json(self, path):
        _path = Path(path)
        
        if not _path.exists():
            raise FileNotFoundError(f'{path} does not exist.')
            
        with open(_path, 'r') as f:
            original_df = json.load(f)
            
        return original_df
    
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
    
    def _get_preprocessing_rule(self, model_nm):
        # create and return a preprocessing rule for a given model
        _drop_text = True
        _make_dummies = True
        
        if model_nm == 'cb':
            # if you want to use text data in the catboost, set _drop_text to False
            _make_dummies = False   # catboost does not need dummies
            
        return _drop_text, _make_dummies    

    def _get_njob_counts(self, model_nm):
        search_n_jobs = 8 if self.n_jobs == -1 else self.n_jobs
        
        if model_nm == 'xgb':
            search_n_jobs = 2
            
        elif model_nm == 'cb':
            search_n_jobs = 1   # intentionally set to 1 because catboost runs really slow if parallel search is executed from the ray
            
        elif model_nm == 'knn':
            search_n_jobs = 2   # too many jobs cause memory error for knn
            
        return search_n_jobs    
    
    def _get_estimator(self, model_nm, param_range, y, is_for_search=True):
        # Although you can manually add parameters by modifying this method, we do not recommend it.
        # One should istead give a dictionary of parameters to the param_range argument of the run method by modifynig the predefined parameter json file.
                
        # check if random state attribute exists in the model class
        if 'random_state' in self.model_tables[model_nm]().get_params().keys():
            model = self.model_tables[model_nm](random_state=self.seed)
            
        else:
            model = self.model_tables[model_nm]()
            
        if model_nm == 'xgb':
            scale_pos_weight = y.value_counts()[1] / (y.value_counts()[0] + y.value_counts()[1])
            
            if self.use_gpu:
                if is_for_search:
                    param_range['tree_method'] = ['gpu_hist']
                    param_range['gpu_id'] = [0]
                    param_range['scale_pos_weight'] = [scale_pos_weight]
                    
                else:
                    param_range['tree_method'] = 'gpu_hist'
                    param_range['gpu_id'] = 0
                    param_range['scale_pos_weight'] = scale_pos_weight

                
        if model_nm == 'cb':
            if is_for_search:
                param_range['max_depth'] = (3, 16)
                param_range['od_wait'] = [10]          
                
            else:
                param_range['od_wait'] = 10

            model.set_params(cat_features=self.cat_cols)
        
        return model
    
    def _search_best_params(self, param_range, skip_param_search, **kwargs):
        model_param_range = self._get_param_ranges(param_range, skip_param_search)
        BEST_PARMAS_PATH = f'{self.base_path}/{self._score_of_interest}/best_params.json'
        
        try:
            self.best_params = self.load_json(BEST_PARMAS_PATH)
            
        except FileNotFoundError:
            self.logger.info(f'{BEST_PARMAS_PATH} does not exist. Start searching best params...')
            self.best_params = {}

        
        for nm, param_range in model_param_range.items():           
            if nm in self.best_params.keys():
                self.logger.info(f'{nm} already has best params. Skip searching best params.')
                continue
            
            if nm in self.skip_model:
                self.logger.info(f'{nm} is in the skip list. Skip searching best params.')
                continue
            
            search_n_jobs = self._get_njob_counts(nm)
            
            self.logger.info(f'Searching best params for {nm} with {search_n_jobs} jobs...')
            
            _drop_text, _make_dummies = self._get_preprocessing_rule(nm)
            X, y, _, _ = self.preprocess(self.original_df, drop_text=_drop_text, make_dummies=_make_dummies)
            # find the best params using only trainin dataset 
            model = self._get_estimator(nm, param_range, y, is_for_search=True)
            
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
                                
            search.fit(X, y)
            
            if self.save_cv_result:
                cv_result = search.cv_results_
                
                if not Path(f'{self.base_path}/cv_result/{self._score_of_interest}').exists():
                    Path(f'{self.base_path}/cv_result/{self._score_of_interest}').mkdir(parents=True)
                    
                pd.DataFrame.from_dict(cv_result).to_csv(f'{self.base_path}/cv_result/{self._score_of_interest}/{nm}.csv', index=False)
            
            self.best_params[nm] = {
                'params': search.best_params_, 
                'score': search.best_score_, 
                'feature_result':
                    {
                        'method': self.feat_selector.get_method_nm(),
                        'selected_features': self.feat_selector.get_feature_names(),
                        'featuer_scores': self.feat_selector.get_feature_score()
                     }
                }
            
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
    
    def _run_predict(self, nm, make_dummies, drop_text, selected_features, **params):
        X, y, x_test, y_test = self.preprocess(self.original_df, make_dummies=make_dummies, drop_text=drop_text, sample=False, select_feature=False, selected_features=selected_features)
        
        model = self._get_estimator(nm, params, y, is_for_search=False)        
        model.set_params(**params)
        
        model.fit(X, y)
        
        proba = model.predict_proba(x_test)[:, 1]
        preds = model.predict(x_test)
        
        # calculate metrics
        try:
            _auc = auc(y_test, proba)
            
        except ValueError as e:
            _auc = f"not able to calculate auc: {e}"
            
        _precision = average_precision_score(y_test, proba, average='micro')
        _recall = recall_score(y_test, preds, average='micro')        
        _f1_score = f1_score(y_test, preds, average='micro')
        
        scores = {'test_auc': _auc, 'test_precision': _precision, 'test_recall': _recall, 'test_f1': _f1_score}
        cv_jobs = self.n_jobs if nm != 'knn' else 1
        
        return scores
    
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
        
        for nm, best_param_result in self.best_params.items():
            params = best_param_result['params']
            selected_feature_lst = best_param_result['feature_result']['selected_features']
            
            if nm in self.model_tables.keys():
                if nm in self.skip_model:
                    self.logger.info(f'{nm} is in the skip list. Skip final evaluation.')
                    continue
                
                if nm in results.keys() and "cv_avg_scores" in results[nm]:
                    # if the result of the model already exists and the values are not related with errors, skip the final evaluation
                    self.logger.info(f'{nm} is in the previous results. Skip final evaluation.')
                    continue
                
                # only run if the model is defined and not in the results
                self.logger.info(f'Running benchmark on {nm} with params {params}')
                
                # get prerprocessing rules
                drop_text, make_dummies = self._get_preprocessing_rule(nm)
                avg_scores = self._run_predict(nm, drop_text=drop_text, make_dummies=make_dummies, selected_features=selected_feature_lst, **params)
                
                results[nm] = {
                    "cv_avg_scores": avg_scores, 
                    "params": params,
                    # 'feature_result':
                    #     {
                    #         'method': self.feat_selector.get_method_nm(),
                    #         'selected_features': self.feat_selector.get_feature_names(),
                    #         'featuer_scores': self.feat_selector.get_feature_score()
                    #     }
                    }
                self.logger.info(f'Finished running benchmark on {nm}')
                
            else:
                results[nm] = 'Undefined'
                    
            gc.collect()
            
            # save intermediate results
            self.save_json(results, self.result_path)
        
        ray.shutdown()
        return results
    

if __name__ == '__main__':
    # create synthethic debug dataset for Benchmark
    import pandas as pd
    import numpy as np
    
    original_df = pd.DataFrame(np.random.randint(0, 100, size=(1000, 4)), columns=list('ABCD'))
    original_df['fradulent'] = np.random.randint(0, 2, size=1000)
    
    # load the predefined parameter ranges
    param_range = "./original_df/param_range.json"
    
    bm = Benchmark(original_df, 
                   label_nm='fradulent', cv=5, 
                   logging_nm='debug')
    
    bm.run(param_range, skip_param_search=False)
    
    