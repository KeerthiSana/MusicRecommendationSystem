import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler

# Loading the saved DBSCAN model
dbscan_model = joblib.load(r'C:\Users\keert\Downloads\MusicRecommedationSystem\dbscan_model.pkl')

# Loading dataset
data = pd.read_csv(r'C:\Users\keert\Downloads\MusicRecommedationSystem\spotify_songs.csv')
#Preprocessing
df=data.dropna()
df = df.drop_duplicates(subset='track_name', keep='first')
df = df.drop(['track_id','playlist_id','track_album_id','duration_ms','track_album_release_date'], axis = 1)
df.reset_index(drop=True, inplace=True)
#encoding
string_columns = ['playlist_name','playlist_genre','playlist_subgenre', 'track_artist', 'track_album_name']
track_names = df['track_name'].tolist()

hasher = FeatureHasher(n_features=1, input_type='string')
hashed_track_name = hasher.fit_transform([[track_name] for track_name in track_names]).toarray()

df['hashed_track_name'] = hashed_track_name

encoder = TargetEncoder(cols=string_columns)
df[string_columns] = encoder.fit_transform(df[string_columns], hashed_track_name)
X_df1 = df.drop(columns=['track_name'])
#scaling
X_df1 = df.drop(columns=['track_name','hashed_track_name'])
scaler = MinMaxScaler()
X_df1_scaled = scaler.fit_transform(X_df1)

# Function to get similar songs using the trained DBSCAN model
def get_similar_songs_dbscan(song_name, df, model, scaled_data=X_df1_scaled, top_n=5):
    if song_name in df['track_name'].values:
        song_index = df[df['track_name'] == song_name].index[0]
        if song_index < len(scaled_data):
            # Extract cluster label of the input song
            song_cluster_label = model.labels_[song_index]
            # Find indices of songs in the same cluster
            cluster_indices = np.where(model.labels_ == song_cluster_label)[0]
            # Filter songs within the same cluster (excluding the input song)
            cluster_df = df.iloc[cluster_indices].drop(song_index)
            
            # Calculate cosine similarity within the cluster
            cosine_similarities = cosine_similarity(scaled_data[song_index].reshape(1, -1), scaled_data[cluster_indices]).flatten()
            
            # Sort songs based on cosine similarity
            similar_songs_indices = np.argsort(cosine_similarities)[::-1][:top_n]
            similar_songs = cluster_df.iloc[similar_songs_indices]
            
            return similar_songs
        else:
            print("Song index is out of bounds.")
            return None
    else:
        print("Song name not found in DataFrame.")
        return None


# Streamlit app code
st.title("Song Recommendation App")

# Dropdown for selecting artist
artist_names = data['track_artist'].unique().tolist()
sorted_artist_names = sorted(map(str, artist_names))
desired_artist = st.selectbox("Select an artist:", [""] + sorted_artist_names)

# Initialize similar songs to an empty DataFrame
similar_songs_dbscan = pd.DataFrame()

# If an artist is selected, filter songs by that artist
if desired_artist:
    # Remove duplicate rows
    data_unique = data.drop_duplicates(subset=['track_name', 'track_artist'])

    # Get songs by the desired artist
    songs_by_artist = data_unique[data_unique['track_artist'] == desired_artist]
    st.subheader(f"Songs by {desired_artist}:")
    for song in songs_by_artist['track_name']:
        st.write(song)

# Get user input for song name
song_name = st.text_input("Enter a song name:")

# If a song name is provided, get similar songs using the trained DBSCAN model
if song_name:
    # Get similar songs using the trained DBSCAN model
    similar_songs_dbscan = get_similar_songs_dbscan(song_name, data, dbscan_model)
    
# Display similar songs
if not similar_songs_dbscan.empty:
    st.subheader("Recommended songs similar to " + song_name)
    for index, row in similar_songs_dbscan.iterrows():
        st.markdown(
            f"""
            <div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 10px;">
                <p style="font-size: 18px; font-weight: bold;">🎵 {row['track_name']}</p>
                <p style="font-size: 16px;">🎤 Artist: {row['track_artist']}</p>
                <p style="font-size: 16px;">📀 Album: {row['track_album_name']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
