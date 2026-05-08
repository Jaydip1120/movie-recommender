import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY="9643C0F1"
#WATCH HISTORY
if "history" not in st.session_state:
st.session_state.history=[]
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# Load Dataset
df = pd.read_csv("final_movies.csv")

# Create Tags
df['tags'] = (
df['genres'].astype(str) + " " +
df['description'].astype(str)
)

# Vectorization
cv = CountVectorizer(
max_features=5000,
stop_words='english'
)
df ['tags']= df['tags'].fillna('')

vectors = cv.fit_transform( df['tags']).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)

# Movie Recommendation Function
def recommend(movie):

movie_index = df[df['title'] == movie].index[0]

distances = similarity[movie_index]

movies_list = sorted(
list(enumerate(distances)),
reverse=True,
key=lambda x:x[1]
)[1:6]
recommended_movies = []

for i in movies_list:
recommended_movies.append(
df.iloc[i[0]].title
)

return recommended_movies
    


# Mood Detection
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


# Mood Recommendation
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

if st.button("Recommend Similar Movies"):
    #save history
  st.session_state.history.append(selected_movie)
    
recommendations = recommend(selected_movie)

  st.subheader("Recommended Movies:")

 for movie in recommendations:
  st.write(movie)

#WATCH HISTORY
st.sidebar.subheader("Watch History")
for movie in
st.session_state.history[-5:]:
st.sidebar.write("Movie")
# Mood Recommendation
st.subheader("😊 Mood Based Recommendation")

selected_mood = st.selectbox(
    "Select Mood",
    ['Comedy', 'Action', 'Romantic', 'Horror', 'General']
)

if st.button("Recommend By Mood"):

    mood_recommendations = recommend_by_mood(selected_mood)

   st.write("Movies For Your Mood:")

    for movie in mood_recommendations:
    st.write(movie)
      
