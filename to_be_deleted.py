import pandas as pd


data = pd.read_csv('./data/fake_job_postings.csv')

def replace_edu(rowdata):
    name = rowdata['required_education']

    if name == 'Some High School Coursework':
        return 'High School or equivalent'
    elif name == 'Vocational - HS Diploma':
        return 'High School or equivalent'
    elif name == 'Vocational - Degree':
        return 'High School or equivalent'
    elif name == 'Vocational':
        return 'High School or equivalent'
    elif name == 'Some College Coursework Completed':
        return "Bachelor's Degree"
    elif name == 'Doctorate':
        return "Professional"
    elif str(name) == 'nan':
        return "Unspecified"
    else:
        return name

data['required_education'] = data.apply(replace_edu, axis=1)

# drop salaray_range
data = data.drop(['salary_range', 'job_id', 'department'], axis=1)


# impute na with "Unsepecified" for all categorical columns
data = data.fillna('Unspecified')
print(data.isna().sum())

data.to_csv('./data/naive_impute/fake_job_postings.csv', index=False)

data2 = pd.read_csv('./data/imputed/words/chained/fake_job_postings.csv')
data3 = pd.read_csv('./data/naive_impute/fake_job_postings.csv')
print(f"data columns: {data.columns}")
print(f"data2 columns: {data2.columns}")
print(f"data3 columns: {data3.columns}")
print(f"xor between data1 and data2: {set(data.columns) ^ set(data2.columns)}")
print(f"xor between data1 and data3: {set(data.columns) ^ set(data3.columns)}")
print(f"xor between data2 and data3: {set(data2.columns) ^ set(data3.columns)}")

