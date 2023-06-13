import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pathlib import Path


TEXT_COL = ['title', 'company_profile', 'description', 'requirements', 'benefits']
URL_COL = ['company_profile', 'description', 'benefits']


def add_rule_url(df):
    url_re = 'https?:\/\/\S+|www\.\S+'

    for col in URL_COL:
        df[f'url_count_{col}'] = df[col].apply(lambda txt: len(re.findall(url_re, txt)) if txt is not np.nan else 0)


def clean_text(text):
    text = text.lower()

    text = text.replace(r'&amp;', '&')
    text = text.replace(r'&nbsp;', ' ')
    text = text.replace(r'&lt;', '<')
    text = text.replace(r'&gt;', '>')
    text = text.replace(r'&quot;', '"')
    text = text.replace(r'\u00a0', ' ')

    text = re.sub('\'re', ' are', text)
    text = re.sub('\'ve', ' have', text)
    text = re.sub('\'m', ' am', text)
    text = re.sub('\'t', ' not', text)
    text = re.sub('\'s', ' ', text)

    sublist = '\[.*?\]|https?:\/\/\S+|www\.\S+|<.*?>+|\n|\w*\d\w*'
    text = re.sub(sublist, '', text)

    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    return text


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keywords_tfidf(vectorizer, feature_names, doc):
    tf_idf_vector = vectorizer.transform([doc])

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items)

    return list(keywords.keys())


def tfidf_stemmed_keywords(DATA, col, topn, target_is_fake=True, sub_na=True, is_stem=True):
    corp = DATA[col][DATA.fraudulent != target_is_fake].tolist()

    sw = stopwords.words('english')
    cv = CountVectorizer(max_df=0.85, stop_words=sw)
    word_cv = cv.fit_transform(corp)

    tfidf_trans = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_trans.fit(word_cv)

    corp_test = DATA[col][DATA.fraudulent == target_is_fake].tolist()

    feature_names = cv.get_feature_names_out()

    k_dict = {}
    for doc in corp_test:
        tfidf_vec = tfidf_trans.transform(cv.transform([doc]))
        sorted_items = sort_coo(tfidf_vec.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        if sub_na and 'unspecified' in keywords:
            continue
        k_dict.update(keywords)

    k_list = list(dict(sorted(k_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:topn]

    if is_stem:
        ss = SnowballStemmer('english')
        k_list = [ss.stem(x) for x in k_list]

    return k_list


def text_c_to_j(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    ss = SnowballStemmer('english')
    stop_words = stopwords.words('english')

    text_list = tokenizer.tokenize(text)
    text_list = [ss.stem(x) for x in text_list if x not in stop_words]
    result = ' '.join(text_list)

    return result


def tfidf_keyword_count(df_c, df_s, col, topn, target_is_fake=True):
    k_list = tfidf_stemmed_keywords(df_c, col, topn, target_is_fake=target_is_fake)

    f_ratio = (df_c['fraudulent'] == 1).sum() / df_c.shape[0]
    r_ratio = (df_c['fraudulent'] == 0).sum() / df_c.shape[0]

    f_count = []
    r_count = []
    for keyword in k_list:
        f_count.append(df_s[col][df_s["fraudulent"] == 1].apply(lambda x: x.count(keyword)).sum())
        r_count.append(df_s[col][df_s["fraudulent"] == 0].apply(lambda x: x.count(keyword)).sum())

    result = pd.DataFrame({'keyword': k_list})
    f_c_ratio = f_count / f_ratio
    r_c_ratio = r_count / r_ratio

    result['fake_count'] = f_c_ratio / (f_c_ratio + r_c_ratio)
    result['real_count'] = r_c_ratio / (f_c_ratio + r_c_ratio)

    return result


def tfidf_select_eff_keywords(df_c, df_s, col, topn, percent=0.9, target_is_fake=True):
    k_count = tfidf_keyword_count(df_c, df_s, col, topn, target_is_fake)
    tar_col = 'fake_count' if target_is_fake else 'real_count'

    eff_k_list = []
    for i, count in enumerate(k_count[tar_col]):
        if count >= percent:
            eff_k_list.append(k_count.keyword[i])

    return eff_k_list


def count_keyword(txt, k_list):
    count = 0
    for keyword in k_list:
        count += txt.count(keyword)
    return count


def add_rule_keywords(df):
    df_cleaned = df.copy(deep=True)
    for col in TEXT_COL:
        df_cleaned[col] = df_cleaned[col].astype(str).apply(lambda x: clean_text(x))

    df_stemmed = df_cleaned.copy(deep=True)
    for col in TEXT_COL:
        df_stemmed[col] = df_stemmed[col].apply(lambda x: text_c_to_j(x))

    for col in TEXT_COL:
        eff_k_list = tfidf_select_eff_keywords(df_cleaned, df_stemmed, col, topn=30, percent=0.8)
        df[f'keyword_count_{col}'] = df_stemmed[col].apply(lambda x: count_keyword(x, eff_k_list))

def generate_features():
    _f_name = 'fake_job_postings.csv'
    _new_f_name = 'fake_job_postings_rule_added.csv'
    _origin_path = f'./data/{_f_name}'

    BASE_DIR = str(Path(__file__).resolve().parent)

    for _chained in ['chained', 'unchained']:
        _path = f'{BASE_DIR}/data/imputed/words/{_chained}/'
        _f_path = _path + _f_name
        
        imputed_df = pd.read_csv(_f_path)
        
        # add 'fraudulent' column from the original if it does not exist
        if 'fradulent' not in imputed_df.columns:
            origin_df = pd.read_csv(_origin_path)
            imputed_df['fraudulent'] = origin_df['fraudulent']

        add_rule_url(imputed_df)
        add_rule_keywords(imputed_df)

        imputed_df.to_csv(_path + _new_f_name, sep=',', na_rep=np.nan, mode='w+')
        
        for col in URL_COL:
            print(f'url_count_{col} nonzero ratio')
            print((imputed_df[f'url_count_{col}'] != 0).sum() / imputed_df.shape[0] * 100)
            print(f'url_count_{col} nonzero and fake ratio')
            print(imputed_df['fraudulent'][imputed_df[f'url_count_{col}'] != 0].sum() / (imputed_df[f'url_count_{col}'] != 0).sum() * 100)
            print()

        for col in TEXT_COL:
            print(f'keyword_count_{col} nonzero ratio')
            print((imputed_df[f'keyword_count_{col}'] != 0).sum() / imputed_df.shape[0] * 100)
            print(f'keyword_count_{col} nonzero and fake ratio')
            print(imputed_df['fraudulent'][imputed_df[f'keyword_count_{col}'] != 0].sum() / (imputed_df[f'keyword_count_{col}'] != 0).sum() * 100)
            print()
            