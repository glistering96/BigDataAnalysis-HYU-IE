import pandas as pd
import numpy as np
from benchmark import Benchmark


def main():    
    np.random.seed(42)
    
    model_param = f"./data/param_range.json"
    
    # load from data/imputed
    for _token in ['chars', 'words']:
        for _chained in ['chained', 'unchained']:
            _path = f'data/imputed/{_token}/{_chained}/fake_job_postings.csv'
            
            # apply one-hot encode
            df = pd.read_csv(_path)
            df = pd.get_dummies(df, columns=['location', 'employment_type', 'required_experience', 'required_education', 
                                             'industry', 'function'])
    
        benchmark = Benchmark(df, 'fraudulent', model_param, f'./data/results/{_token}/{_chained}/')
        results = benchmark.run(model_param, skip_param_search=False)
        print(results)


if __name__ == '__main__':
    main()