""" set a range of parameters for each model according to the model table from Benchmark class

    model_tables = {'rf': RandomForestClassifier, 
                    'lr': LogisticRegression,
                    'svm': SVC,
                    'knn': KNeighborsClassifier,
                    'ada': AdaBoostClassifier,
                    'gb': GradientBoostingClassifier,
                    'et': ExtraTreesClassifier,
                    'xgb': XGBClassifier
                    }
"""

params = {
    'rf': {'n_estimators': [30, 50, 100, 200],
              'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [2, 3, 4, 5],
                'criterion': ['gini', 'entropy']
                },
    
    'lr': {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'solver': ['lbfgs'],
    },
    
    'svm': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'kernel': ['rbf'],
                
    },
    
    'knn': {'n_neighbors': [3, 5, 7, 9],
    },
    
    'ada': {'n_estimators': [30, 50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1]
    },
    
    'gb': {'n_estimators': [30, 50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [2, 3, 4, 5],
    },
    
    'et': {'n_estimators': [30, 50, 100, 200],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [2, 3, 4, 5],
                
    },
    
    'xgb': {'n_estimators': [30, 50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [2, 3, 4, 5],
                
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
    