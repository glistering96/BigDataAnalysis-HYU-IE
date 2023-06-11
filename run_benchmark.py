import pandas as pd
import numpy as np
from benchmark import Benchmark

# filter UndefinedMetricWarning:
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
                                    logging_nm= f'{_token}_{_chained}_{rule_str}_ray',
                                        label_nm='fraudulent')
        
                results = benchmark.run(model_param, skip_param_search=False)
                print(results)


if __name__ == '__main__':
    main()