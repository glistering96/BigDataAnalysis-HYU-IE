import pandas as pd
import numpy as np
from benchmark import Benchmark


# select one of methods in [None, random_over, random_under, smote] in the below variable.
# adasyn, cluster_centroid is available but we do not consider these two here

_sample_method = 'random_under' # str type is required

def main():   
    model_param = f"./data/param_range.json"
    
    # load from data/imputed
    for _token in ['words']:
        for _chained in ['chained', 'unchained']:
            for _rule in [False]:
                
                _path = f'./data/imputed/{_token}/{_chained}/fake_job_postings.csv'
                _original_path = f'./data/fake_job_postings.csv'
                
                # apply one-hot encode
                imputed = pd.read_csv(_path)
                original = pd.read_csv(_original_path)
                rule_str  = 'rule' if _rule else 'no_rule'
                
                benchmark = Benchmark(imputed, original,
                                      sample_method=_sample_method, 
                                      logging_nm= f'{_token}_{_chained}_{_sample_method}',
                                      label_nm='fraudulent')
        
                results = benchmark.run(model_param, skip_param_search=False)
                print(results)


if __name__ == '__main__':
    main()