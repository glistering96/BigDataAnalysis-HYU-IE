import pandas as pd
import warnings
import datawig
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings('ignore')


df = pd.read_csv(f'{BASEDIR}/data/fake_job_postings.csv', engine = 'python')
df = df.drop(['job_id'], axis = 1)

# 우선 text 데이터 제외한 예측
df = df.drop(['title','company_profile','description','requirements','benefits'], axis = 1)

# 대부분 missing 인 col 제거
df = df.drop(['department','salary_range'], axis = 1)
df.head()

# location 전처리

print('location null의 fake job posting 비율 : ', df[df['location'].isnull() ==True]['fraudulent'].value_counts()[1]/df[df['location'].isnull() ==True].shape[0])

# 국가명만 남기기
df['location'] = df['location'].apply(lambda x : str(x).split(',')[0])

# mode로 대체
df[df['location'].isnull() ==True]['location'] = df['location'].mode()[0]
# 10개 이상의 관측치 보유 location 만 남기기
loc_val_count = df['location'].value_counts()
loc_over_10 = loc_val_count[loc_val_count>10].index
df = df[df['location'].isin(loc_over_10)].reset_index(drop=True)

print("비율 차이 적으므로 mode로 imputation")


def impute_train_whole(df, target_col_nm, token, chained):
    input_columns = list(set(df.columns) - set(target_col_nm))

    imputer = datawig.SimpleImputer(input_columns=input_columns,
                                    output_column=target_col_nm,
                                    tokens=token,
                                    output_path=f'{BASEDIR}/imputer/{token}/{chained}/{target_col_nm}'
                                    )

    imputer.fit(train_df=df, num_epochs=100, batch_size=128)

    imputed = imputer.predict(df)

    return pd.DataFrame(imputed)


def run_impute(df, token, chained):
    target_col_nms = df.columns[df.isnull().any()].value_counts().sort_values(ascending=True).index.tolist()
    impute_result = {}
    temp = df.copy(deep=True)
    chained_nm = 'chained' if chained else 'unchained'

    for target_col_nm in target_col_nms:
        if chained:
            temp = df.copy(deep=True)

        print(f'Imputing {target_col_nm}...')
        imputed = impute_train_whole(temp, target_col_nm, token, chained_nm)
        print(f'Imputed {target_col_nm}...')

        impute_result[target_col_nm] = imputed

    final_df = df.copy(deep=True)

    for k, v in impute_result.items():
        final_df[k] = v[k]

    path = f'../data/imputed/{token}/{chained_nm}'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    final_df.to_csv(f"{path}/fake_job_postings.csv'", index=False)

    return final_df


for chained in [True, False]:
    for token in ['chars', 'words']:
        run_impute(df, token, chained)