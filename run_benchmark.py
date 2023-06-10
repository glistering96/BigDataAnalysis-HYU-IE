import pandas as pd
import numpy as np
from benchmark import Benchmark

# filter UndefinedMetricWarning:
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess(imputed_df, original_df):
    df = imputed_df.copy(deep=True)
    # drop text cols
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df = df.drop(text_cols, axis=1)
    
    # concat fradulent col to imputed_df from original_df
    df['fraudulent'] = original_df['fraudulent']
    
    # integerate some values that have too low frequency
    return df


def main():   
    model_param = f"./data/param_range.json"
    
    # load from data/imputed
    for _token in ['words']:
        for _chained in ['chained', 'unchained']:
            for _rule in [False]:
                
                _path = f'./data/imputed/{_token}/{_chained}/fake_job_postings.csv'
                _original_path = f'./data/fake_job_postings.csv'
                
                # apply one-hot encode
                df = pd.read_csv(_path)
                original = pd.read_csv(_original_path)
                df = preprocess(df, original)
                df = pd.get_dummies(df, columns=['location', 'employment_type', 'required_experience', 'required_education', 
                                                'industry', 'function'])
        rule_str  = 'rule' if _rule else 'no_rule'
        benchmark = Benchmark(df,
                              logging_nm= f'{_token}_{_chained}_{rule_str}',
                                label_nm='fraudulent')
        
        results = benchmark.run(model_param, skip_param_search=False)
        print(results)


if __name__ == '__main__':
    main()