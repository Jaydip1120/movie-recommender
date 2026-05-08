import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "9643cf01"

# WATCH HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# Background Image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    h1, h2, h3, p, label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Dataset
df = pd.read_csv("final_movies.csv")

# Create Tags
df['tags'] = (
    df['genres'].astype(str) + " " +
    df['description'].astype(str)
)

# Fill Missing Values
df['tags'] = df['tags'].fillna('')

# Vectorization
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(df['tags']).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)


# Fetch Poster from OMDb API
def fetch_poster(movie_title):

    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        if data.get('Response') == 'True':

            poster = data.get('Poster')

            # Check valid poster
            if poster and poster != "N/A":
                return poster

    except:
        pass

    # Default Poster
    return "https://via.placeholder.com/300x450.png?text=No+Poster"


# Movie Recommendation Function
def recommend(movie):

    movie_index = df[df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:

        movie_title = df.iloc[i[0]].title

        recommended_movies.append(movie_title)

        recommended_posters.append(
            fetch_poster(movie_title)
        )

    return recommended_movies, recommended_posters


# Mood Detection Function
def get_mood(genres):

    genres = str(genres).lower()

    if 'comedy' in genres:
        return 'Comedy'

    elif 'action' in genres:
        return 'Action'

    elif 'romance' in genres:
        return 'Romantic'

    elif 'horror' in genres:
        return 'Horror'

    else:
        return 'General'


# Create Mood Column
df['mood'] = df['genres'].apply(get_mood)


# Mood Recommendation Function
def recommend_by_mood(mood):

    mood_movies = df[
        df['mood'].str.lower() == mood.lower()
    ]

    recommendations = mood_movies.sample(2)

    return recommendations['title'].values


# STREAMLIT UI
st.title("🎬 Movie Recommendation System")


# Movie Search
selected_movie = st.selectbox(
    "Search Movie",
    df['title'].values
)


# Recommend Similar Movies
if st.button("Recommend Similar Movies"):

    # Save History
    st.session_state.history.append(selected_movie)

    names, posters = recommend(selected_movie)

    st.subheader("Recommended Movies")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0], width=150)
        st.write(names[0])

    with col2:
        st.image(posters[1], width=150)
        st.write(names[1])

    with col3:
        st.image(posters[2], width=150)
        st.write(names[2])

    with col4:
        st.image(posters[3], width=150)
        st.write(names[3])

    with col5:
        st.image(posters[4], width=150)
        st.write(names[4])


# WATCH HISTORY
st.sidebar.subheader("Watch History")

for movie in st.session_state.history[-5:]:
    st.sidebar.write(movie)


# Mood Recommendation
st.subheader("😊 Mood Based Recommendation")

selected_mood = st.selectbox(
    "Select Mood",
    ['Comedy', 'Action', 'Romantic', 'Horror', 'General']
)


if st.button("Recommend By Mood"):

    mood_recommendations = recommend_by_mood(selected_mood)

    st.write("Movies For Your Mood:")

    cols = st.columns(len(mood_recommendations))

    for idx, movie in enumerate(mood_recommendations):

        poster = fetch_poster(movie)

        with cols[idx]:
            st.image(poster, width=180)
            st.write(movie)