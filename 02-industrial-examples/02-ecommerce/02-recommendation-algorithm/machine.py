import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

def preprocess():

    netflix_overall=pd.read_csv("data/netflix_titles.csv")

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
    
    return netflix_overall, cosine_sim, indices
    
    
def get_recommendations_new(title, netflix_overall, cosine_sim, indices):
    
    pd.set_option('display.max_columns', None)
    title=title.replace(' ','').lower()
    
    try:
        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:11]

        movie_indices = [i[0] for i in sim_scores]

        recomendation = netflix_overall[['title','country','release_year']].iloc[movie_indices]
        
        sim_pd = pd.DataFrame(sim_scores)[[1]]
        
        sim_pd = sim_pd.rename(columns={1:'Similiarity'})

        recomendation = recomendation.reset_index(drop=True)

        recomendation = pd.concat([recomendation, sim_pd], axis=1)
    
        recomendation.index += 1
        
        
        return print(recomendation)
    
    except:
        print("오류 : 올바른 title 명을 적어 주세요.")
