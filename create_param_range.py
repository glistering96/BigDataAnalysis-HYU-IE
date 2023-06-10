""" set a range of parameters for each model according to the model table from Benchmark class

    model_tables = {'rf': RandomForestClassifier, 
                    'lr': LogisticRegression,
                    'svm': SVC,
                    'knn': KNeighborsClassifier,
                    'ada': AdaBoostClassifier,
                    'gb': GradientBoostingClassifier,
                    'et': ExtraTreesClassifier,
                    'xgb': XGBClassifier,
                    'cb: CatBoostClassifier
                    }
"""
N_ESTIAMTORS =  [100, 500, 1000]
LR = [0.001, 0.01, 0.1]
MAX_DEPTH = [6, 9, 15, 20, 30]
MIN_SAMPLES_SPLIT = [2, 5, 10, 15]
MIN_SAMPLES_LEAF = [2, 5, 10, 15]

params = {
    'rf': {'n_estimators': N_ESTIAMTORS,
              'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                'criterion': ['gini', 'entropy']
                },
    
    'lr': {'penalty': ['l1', 'l2', 'elasticnet', None],
                'C': [0.001,  0.1, 1, 10, 100],
                'solver': ['newton-cholesky'],
                'max_iter': [50000]
    },
    
    'svm': {'C': [0.001, 0.01, 0.1, 1, 10, 50, 100],
                'kernel': ['rbf'],
                
    },
    
    'knn': {'n_neighbors': [15, 17, 20, 25, 30],
    },
    
    'ada': {'n_estimators': N_ESTIAMTORS,
                'learning_rate': LR
    },
    
    'gb': {'n_estimators': N_ESTIAMTORS,
                'learning_rate': LR,
                'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
    },
    
    'et': {'n_estimators': N_ESTIAMTORS,
                'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                
    },
    
    'xgb': {'n_estimators': N_ESTIAMTORS,
                'learning_rate': LR,
                'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                
    },
    
    'cb': {'n_estimators': N_ESTIAMTORS,
                   'learning_rate': LR,
                    'max_depth': MAX_DEPTH,
                    'min_samples_split': MIN_SAMPLES_SPLIT,
                    'min_samples_leaf': MIN_SAMPLES_LEAF,
                    
        }        
    
}

import json
from pathlib import Path

# save the params to json file
_path = Path('./data/param_range.json')

if not _path.parent.exists():
    _path.parent.mkdir(parents=True)
    
with open(_path, 'w') as f:
    json.dump(params, f, indent=4)
    