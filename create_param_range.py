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
LR = (0.001, 0.3)
MAX_DEPTH = (6, 50)
MIN_SAMPLES_SPLIT = (2, 30)
MIN_SAMPLES_LEAF = (2, 30)
C = (0.001, 100)
N_NEIGHBORS = (3, 100)
REG = (0.001, 0.1)
MIN_CHILD_WEIGHT = (0.001, 100)
GAMMA = (0.001, 100)


params = {
    'rf': {'n_estimators': N_ESTIAMTORS,
              'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                'criterion': ['entropy']
                },
    
    'lr': {'penalty': ['l2', None],
                'C': C,
                'solver': ['newton-cholesky'],
                'max_iter': [100000]
    },
    
    'svm': {'C': C,
                'kernel': ['rbf'],
                
    },
    
    'knn': {'n_neighbors': N_NEIGHBORS,
    },
    
    'ada': {'n_estimators': N_ESTIAMTORS,
                'learning_rate': LR
    },
    
    # 'gb': {'n_estimators': N_ESTIAMTORS,
    #             'learning_rate': LR,
    #             'max_depth': MAX_DEPTH,
    #             'min_samples_split': MIN_SAMPLES_SPLIT,
    #             'min_samples_leaf': MIN_SAMPLES_LEAF,
    # },
    
    'et': {'n_estimators': N_ESTIAMTORS,
                'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                
    },
    
    'xgb': {'n_estimators': N_ESTIAMTORS,
                'learning_rate': LR,
                'max_depth': MAX_DEPTH,
                'min_child_weight': MIN_SAMPLES_SPLIT,
                'reg_alpha': REG,
                'reg_lambda': REG,
                'gamma': GAMMA,
                
    },
    
    'cb': {'iterations': N_ESTIAMTORS,
                   'learning_rate': LR,
                    'max_depth': MAX_DEPTH,
                    'l2_leaf_reg': REG,
                    'auto_class_weights': ['Balanced'],
                    'boosting_type': ['Ordered'],
                    'min_data_in_leaf': MIN_SAMPLES_LEAF,
                    
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
    