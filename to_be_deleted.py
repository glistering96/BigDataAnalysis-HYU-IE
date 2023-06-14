import pandas as pd


data = pd.read_csv('./data/fake_job_postings.csv')

# drop salaray_range
data = data.drop(['salary_range'], axis=1)


# impute na with "Unsepecified" for all categorical columns
data = data.fillna('Unspecified')
print(data.isna().sum())

data.to_csv('./data/naive_impute/fake_job_postings.csv', index=False)