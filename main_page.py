import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_page import run_recommender_1,check_availability

# def load_data():
#     with open('saved_model.pk1','rb') as file:
#         data=pickle.load(file)
#     return data

# data=load_data()

# cv_loaded=data['cv']
# cm=data['count_matrix']
# cs=data['cosine_sim']


def show_main_page():
    st.title("Movie Recommendation System")

    st.write("""### Tell us the movies you loved watching and we will suggest you some that you might like""")
    
    movie_liked = st.text_input("Enter a movie name")

    ok=st.button("Show Movies")

    if ok:
        if check_availability(movie_liked):
            run_recommender_1(movie_liked)
        else:
            
            st.subheader("Sorry, this movie does not exist in our database")
           


  