import pandas as pd
import numpy as np
from src.benchmark import Benchmark
from itertools import product

# select one of methods in [None, random_over, random_under, smote] in the below variable.
# adasyn, cluster_centroid is available but we do not consider these two here

_sample_method = None # str type is required

# SMOTE: yejoon
# random_over: wonjun
# random_under: woochan

def main():   
    model_param = f"./data/param_range.json"
    _token = 'words'

    _chained_lst = ['chained', 'unchained']
    _rule_lst = [False, True]
    _feature_select_lst = ['mutual_info_classif', 'chi2']
    
    # get the combination of chained, rule, feature_select
    
    for _chained, _rule, _feature_select in product(_chained_lst, _rule_lst, _feature_select_lst):
        _rule_txt = '_rule_added' if _rule else ''
        _path = f'./data/naive_impute/fake_job_postings.csv'
        
        # read csv and pass it 
        df = pd.read_csv(_path)
        
        benchmark = Benchmark(df,
                                sample_method=_sample_method, 
                                logging_nm= f'baseline_no_imputation_{_feature_select}',
                                feat_select_method=_feature_select,
                                label_nm='fraudulent',
                                skip_model=['cb']
        )

        results = benchmark.run(model_param, skip_param_search=False)

if __name__ == '__main__':
    main()