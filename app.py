import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

url='https://raw.githubusercontent.com/JoelAnalytics/beer_recommender/main/data/beer_data_set.csv'
header=st.container()
features=st.container()

df=pd.read_csv(url)
df_details=df['Style']+' '+df['Description']



lemmatizer = WordNetLemmatizer()

def normalize_text(y):
    normalized_y=nltk.word_tokenize(y)
    normalized_y=[lemmatizer.lemmatize(word) for word in normalized_y]
    sentence= ' '.join(normalized_y)
    return sentence

def get_similarities(X,y):
    vectorized_text = vectorizer.transform([y])
    similarities = cosine_similarity(X, vectorized_text)[:,0]
    return similarities


for x in range(len(df_details)):
    df_details[x]=normalize_text(df_details[x])


vectorizer= TfidfVectorizer()
X=vectorizer.fit_transform(df_details)

with header:
    st.title('Welcome to Joe Bar')
    st.markdown('**We would like to suggest you a new beer today**')


with features:
    st.header('Lets find a new beer for you')

    sel_col, disp_col = st.columns(2)
    input_feature=sel_col.text_input('Which kind of beer would you like to drink? :smile:','')

    result=st.button('FIND')
    if result:
        similarities=get_similarities(X,input_feature)
        index_top_bier = np.argmax(similarities)
        st.write(f'I recommend you take beer: {df_details.iloc[index_top_bier]}')



