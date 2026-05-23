import streamlit as st
import pickle

movies = pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x:x[1])[1:11]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


st.title("Movie Recommender System")

movies_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a movie",
    movies_list
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    for movie in recommendations:
        st.write(movie)