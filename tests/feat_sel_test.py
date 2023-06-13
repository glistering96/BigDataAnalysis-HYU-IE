import sys
from pathlib import Path

basedir = str(Path(__file__).resolve().parents[1])

sys.path.append(basedir)

import pandas as pd
import numpy as np
from src.benchmark import Benchmark


def method_test():
    """
    create a radom dataset following this format:
                imputed = pd.read_csv(_path)
                original = pd.read_csv(_original_path)
                rule_str  = 'rule' if _rule else 'no_rule'
                
                benchmark = Benchmark(imputed, original,
                                      sample_method=_sample_method, 
                                      logging_nm= f'{_token}_{_chained}_{_sample_method}',
                                      label_nm='fraudulent')
        
                results = benchmark.run(model_param, skip_param_search=False)
    """
    
    imputed = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    
    # for original data, it should include a binary label column named with 'fraudulent'
    original = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    original = original.assign(fraudulent = np.random.randint(0,2,size=(100, 1)))
    
    
    feat_sel_method = 'f_classif'
    feat_sel_method = 'mutual_info_classif'
    feat_sel_method = 'chi2'
    
    benchmark = Benchmark(imputed, original,
                          sample_method=None,
                            logging_nm= f'test',
                            feat_select_method=feat_sel_method,
                            text_cols=[],
                            cat_cols=[],
                            label_nm='fraudulent')
    
    result = benchmark.run('./data/param_range.json', skip_param_search=False)


if __name__ == '__main__':
    method_test()
    