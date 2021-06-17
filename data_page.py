import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def get_title_from_index(index,df):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title,df):
    return df[df.title == title]["index"].values[0]



def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print ("Error:",row)

def load_df():
    dataf=pd.read_csv("movie_dataset.csv")
    return dataf

def run_recommender_1(movie_user_likes):
    df=load_df()
    
    features=['keywords','cast','genres','director']

    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"]=df.apply(combine_features,axis=1)
    cv = CountVectorizer()
    count_matrix=cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)

    movie_index = get_index_from_title(movie_user_likes,df)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x: x[1],reverse=True)
    i=0
    for movie in sorted_similar_movies:
        if i!=0:
            st.write(get_title_from_index(movie[0],df))
            
        i+=1
        if i>50:
            break

def check_availability(movie_name):
    df=load_df()
    for names in df['original_title']:
        if movie_name == names:
            return True
    return False