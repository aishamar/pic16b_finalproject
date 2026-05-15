#!/usr/bin/env python
# coding: utf-8
"""
Music Recommendor Algorithm
Collaborators: Aisha Mardini, Lilly Peters, Shinung Li

First dataset:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

- used for content based filtering

Elements
--------
track_id: The Spotify ID for the song
artists: The artists' names. If there is more than one artist, they are separated by a ;
album_name: The album name
track_name: Name of the track
popularity: The popularity is calculated by Spotify's algorithm from [0,100]. Based on the total number of plays the track has had and how recent those plays are.
duration_ms: The song length in milliseconds
explicit: Whether or not the song has explicit lyrics (true = yes; false = no OR unknown)
danceability: How suitable a track is for dancing from [0,1] based on a combination of elements including tempo, rhythm stability, beat strength, and overall regularity.
energy: Represents a perceptual measure of intensity and activity from [0,1]. Typically, energetic tracks feel fast, loud, and noisy.
key: The key the track is in mapped using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: How loud the track is in decibels
mode: Mode indicates the modality (major or minor) of a trackMajor is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track from [0,1]
acousticness: Whether the track is acoustic from [0,1]. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals from [0,1]
liveness: Detects the presence of an audience in the recording from [0,1]
valence: Musical positiveness conveyed by a track from [0,1]. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
tempo: Estimated tempo of a track in beats per minute (BPM).
time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
track_genre: The genre of the song


"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1:
#Upload the data
#Data cleaning and preprocessing

# Loading in Spotify datasets
df_read = pd.read_csv('/Users/aishamardini/pic16b_finalproject/dataset.csv')
print(df_read.head())

# checking if there are NaN values in track id, artists, albums
print(df_read[df_read["track_id"].isna()])

print(df_read[df_read["artists"].isna()])

print(df_read[df_read["album_name"].isna()])

# drop nan values
df_read = df_read.dropna()

# check to see if any rows are duplicates and drop them
#print(df_read.drop_duplicates().any())
df_read = df_read.drop_duplicates()

# checking the types for columns
print(df_read.dtypes)

# check column names
print(df_read.columns)

# drop the unnamed column
df = df_read.drop(columns=['Unnamed: 0'])

# check statistics
print(df.describe())

# check to see if multiartists are separated by semicolon
df[df["artists"].str.contains(";")]["artists"].unique()[:20]

# how many loudness columns are 0
print((df["loudness"] == 0).sum())

# how many instrumentalness columns are 0
print((df["instrumentalness"] == 0).sum())

# how many tempo columns are 0
print((df["tempo"] == 0).sum())

# drop temp columns that are 0
df = df[df["tempo"] != 0]

# remove artist's duplicated songs
df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"], keep="first")

# check to see dropped
print(df.duplicated(subset=["track_name", "artists"]).sum())

# remove songs with popularity == 0
df = df[df["popularity"] > 0]

# check statistics again
print(df.describe())



# Step 2: Data Visualization

# bar plot of the first 15 most popular genres

mostpop = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(15)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_title("Top 15 Popular Genres")
sns.barplot(x=mostpop.index, y=mostpop.values, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# plot for the 15 least popular genres

leastpop = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=True).head(15)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_title("15 Least Popular Genres")
sns.barplot(x=leastpop.index, y=leastpop.values, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.show()

# heatmap plot correlating features
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".1f", ax=ax3)
ax3.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 5 density plots comparing features with energy
# way to see how well songs can be correlated

features = ["valence", "loudness", "tempo", "acousticness"]

for f in features:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="energy", y=f, fill=True, levels=10)
    plt.title(f"Density of Energy vs {f}")
    plt.tight_layout()
    plt.show()


""" 
Second Dataset:
https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data

- used for collaborative filtering

Elements
--------
msno: user id
song_id: song id
source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
source_screen_name: name of the layout a user sees.
source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc.
target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise.

"""
"""
train = pd.read_csv('/Users/aishamardini/pic16b_finalproject/train.csv')
train.columns
print(train.duplicated().sum())
train.isnull().sum()
train = train[['msno', 'song_id', 'source_system_tab', 'source_type', 'target']]
train

train['source_system_tab'].unique()
train['source_system_tab'].value_counts()
train = train[~train['source_system_tab'].isin(['notification', 'settings'])]
train

songs = pd.read_csv('/Users/lillypeters/Downloads/songs.csv')



members = pd.read_csv('/Users/aishamardini/pic16b_finalproject/members.csv')
members = members[['bd', 'msno', 'city']]
members.dropna()

song_extra = pd.read_csv('/Users/aishamardini/pic16b_finalproject/song_extra_info.csv')
song_extra

merged_song = pd.merge(songs, song_extra[['song_id', 'name']], on='song_id', how='left')

merged_song.dropna()



merged_song.dropna()

merged_song = merged_song[['artist_name', 'name', 'song_id']]
merged_song


merged_song[merged_song.duplicated()]
print(merged_song.duplicated().sum())



merged_song = merged_song.drop_duplicates()


colab_df= pd.merge(train, merged_song[['song_id', 'name', 'artist_name']], on='song_id', how='left')

colab_df

kaggle = pd.merge(colab_df, members[['bd', 'msno', 'city']], on='msno', how='left')

kaggle.rename(columns={'bd': 'age'})

kaggle['source_system_tab'].unique()
kaggle = kaggle.dropna(subset=['source_system_tab'])
kaggle['source_system_tab'].unique()
kaggle['source_system_tab'] = kaggle['source_system_tab'].map({
    'my library': 6,    # strongest signal
    'search': 5,        # user actively looked for it
    'discover': 4,      
    'explore': 3,
    'radio': 2,
    'listen with': 1    # weakest signal
})


kaggle['source_system_tab'].unique()
kaggle['source_system_tab'].unique()
kaggle['source_system_tab'].value_counts()
"""
