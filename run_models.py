# Predictor model: Random Forest, XGBoost, CatBoost, SVM, Logistic Regression, KNN ...?

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.exceptions import FitFailedWarning

from xgboost import XGBClassifier

import json
from pathlib import Path
from src.common.logger import init_logger


class Benchmark:
    model_tables = {'rf': RandomForestClassifier, 
                    'lr': LogisticRegression,
                    'svm': SVC,
                    'knn': KNeighborsClassifier,
                    'ada': AdaBoostClassifier,
                    'gb': GradientBoostingClassifier,
                    'et': ExtraTreesClassifier,
                    'xgb': XGBClassifier
                    }
    
    BASEDIR = str(Path(__file__).parent)
    
    def __init__(self, 
                 data,
                 logging_nm='benchmark', 
                 label_nm='fradulent',
                 scoring={'precision': 'precision', 'recall': 'recall', 'f1': 'f1'},
                 cv=10,
                 seed=42
                 ) -> None:
        self.data = data
        self.label_nm = label_nm
        self.cv = cv
        self.result_path = f'{self.BASEDIR}/{logging_nm}/results.json'
        self.scoring = scoring
        self.logger = init_logger({'level': 'INFO', 'name': logging_nm})
        self.seed = seed
    
    def _run_cv(self, model, **kwargs):
        y = self.data[self.label_nm]
        X = self.data.drop(self.label_nm, axis=1)
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        
        try:
            cv_scores = cross_validate(model(**kwargs), X, y, cv=cv, scoring=self.scoring)
            
        except FitFailedWarning as e:
            self.logger.error(f'FitFailedWarning: {e}')
            raise Exception(f'FitFailedWarning: {e}')
        
        return cv_scores
    
    def save_json(self, data, path):
        _path = Path(path)
        
        if not _path.parent.exists():
            _path.parent.mkdir(parents=True)
            
        with open(_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_json(self, path):
        _path = Path(path)
        
        if not _path.exists():
            raise FileNotFoundError(f'{path} does not exist.')
            
        with open(_path, 'r') as f:
            data = json.load(f)
            
        return data        
    
    def run(self, models: dict):
        # models: {model_nm: model_params, ...}
        
        try:
            results = self.load_json(f'{self.BASEDIR}/resresults.json')
            self.logger.info('Previous results found. Running benchmark on the loaded result set...')
            
        except FileNotFoundError:
            self.logger.info('No previous results found. Running benchmark on a new result set...')
            results = {}
        
        for nm, params in models.items():
            try:
                if nm in self.model_tables.keys():
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
            
        return results
    

if __name__ == '__main__':
    # create a random dataset for debugging
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    data = pd.DataFrame(np.random.randint(0, 1000, size=(1000, 4)), columns=list('ABCD'))
    data['fradulent'] = np.random.randint(0, 2, size=(1000, 1))
    
    benchmark = Benchmark(data, logging_nm='debug')
    
    models = {
        'rf': {'random_state': 42},
        'lr': {'random_state': 42},
        'svm': {'random_state': 42},
        'knn': {'n_neighbors': 5},
        'ada': {'random_state': 42},
        'gb': {'random_state': 42},
        'et': {'random_state': 42},
        'xgb': {'random_state': 42}
    }
    
    results = benchmark.run(models, )
    print(results)