import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "2e3151552627ca728fef1c002b5f401c"
st.set_page_config(layout="wide")

st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(to bottom right, #0f0f0f, #1a1a2e);
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* Main title */
.main-title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:white;
    margin-bottom:0;
}

.sub-title{
    text-align:center;
    color:#aaaaaa;
    font-size:20px;
    margin-top:0;
    margin-bottom:40px;
}

/* Recommendation button */
.stButton>button{
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color:white;
    border:none;
    border-radius:15px;
    padding:12px 28px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
    box-shadow:0 4px 15px rgba(255,75,43,0.4);
}

/* Button hover */
.stButton>button:hover{
    transform:scale(1.05);
    box-shadow:0 6px 25px rgba(255,75,43,0.6);
}

/* Movie cards */
.movie-card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius:20px;
    padding:15px;
    text-align:center;
    transition:0.3s;
    margin-bottom:20px;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
}

/* Hover effect */
.movie-card:hover{
    transform: translateY(-10px) scale(1.03);
    box-shadow:0 10px 30px rgba(255,255,255,0.15);
}

/* Movie titles */
.movie-title{
    font-size:18px;
    font-weight:bold;
    min-height:60px;
    color:white;
    margin-bottom:10px;
}

/* Selectbox */
.stSelectbox label{
    font-size:18px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        poster_path = data.get('poster_path')

        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        else:
            return "https://dummyimage.com/300x450/000/fff&text=No+Poster"

    except:
        return "https://dummyimage.com/300x450/000/fff&text=No+Poster"

movies_dict = pickle.load(open('movies_list.pkl','rb'))
movies = pd.DataFrame(movies_dict)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x:x[1])[1:11]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list[:5]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
       

    return recommended_movies, recommended_movies_posters


st.markdown("""
<h1 class='main-title'>Movie Recommender System</h1>
<p class='sub-title'>Discover your next favorite movie 🍿</p>
""", unsafe_allow_html=True)

movies_list = movies['title'].values

selected_movie = st.selectbox(
    "Type or select a movie",
    movies['title'].values
)

if st.button('Recommend'):

    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for idx, col in enumerate(cols):

        with col:

            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">
                    {names[idx]}
                </div>
            """, unsafe_allow_html=True)

            poster = posters[idx]

            if poster and str(poster).startswith("http"):
                st.image(poster, use_container_width=True)
            else:
                st.image(
                    "https://dummyimage.com/300x450/000/fff&text=No+Poster",
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
