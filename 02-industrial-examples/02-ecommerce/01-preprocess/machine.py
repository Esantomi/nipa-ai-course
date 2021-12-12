import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from tqdm import trange

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

def preprocess():

    pd.set_option('display.max_columns', None)
    print("데이터 읽는 중...")
    netflix_overall=pd.read_csv("data/netflix_titles.csv")

    print("읽어 온 데이터를 출력합니다.")
    print(netflix_overall)
    
    filledna=netflix_overall.fillna('')

    features=['title','director','cast','listed_in','description']
    filledna=filledna[features]

    for feature in features:
        filledna[feature] = filledna[feature].apply(clean_data)

    filledna['soup'] = filledna.apply(create_soup, axis=1)
    
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(filledna['soup'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    filledna=filledna.reset_index()
    indices = pd.Series(filledna.index, index=filledna['title'])
    
    print("학습을 수행합니다.")
    
    for j in trange(20,file=sys.stdout, leave=False, unit_scale=True, desc='학습 진행률'):
        
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    print('학습이 완료되었습니다.')
    return netflix_overall, cosine_sim, indices
    