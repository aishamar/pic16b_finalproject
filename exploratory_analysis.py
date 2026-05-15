#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/lillypeters/Downloads/dataset.csv')
df


# In[3]:


df[df["track_id"].isna()]


# In[4]:


df[df["artists"].isna()]


# In[5]:


df[df["album_name"].isna()]


# In[6]:


df = df.dropna()


# In[7]:


df = df.drop_duplicates()


# In[8]:


df.dtypes


# In[9]:


df.describe()
df.head()


# In[10]:


df.columns


# In[11]:


df[df["artists"].str.contains(";")]["artists"].unique()[:20]


# In[12]:


df.describe()


# In[13]:


(df["loudness"] == 0).sum()


# In[14]:


(df["tempo"] == 0).sum()


# In[15]:


df = df[df["tempo"] != 0]


# In[16]:


df.describe()


# In[17]:


df = df.drop(columns=["Unnamed: 0"])


# In[18]:


df.describe()


# In[19]:


(df["instrumentalness"] == 0).sum()


# In[20]:


#Exploratory Data Analysis


# In[21]:


df = df.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"], keep="first")


# In[22]:


sns.barplot(df, x = "track_genre", y="popularity")

plt.show()


# In[23]:


df["track_genre"].count()


# In[24]:


genre_pop = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(15)
sns.barplot(x=genre_pop.index, y=genre_pop.values)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[25]:


genre_pop = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=True).head(15)
sns.barplot(x=genre_pop.index, y=genre_pop.values)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[26]:


sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".1f")
plt.show()


# In[27]:


sns.histplot(df["popularity"], bins=30)
plt.show()


# In[28]:


df = df[df["popularity"] > 0]


# In[29]:


sns.histplot(df["popularity"], bins=30)
plt.savefig('popularity.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:





# In[30]:


df.duplicated(subset=["track_name", "artists"]).sum()


# In[ ]:





# In[31]:


df.duplicated(subset=["track_name", "artists"]).sum()


# In[237]:


train


# In[238]:


train = pd.read_csv('/Users/lillypeters/Downloads/train.csv')
train.columns
print(train.duplicated().sum())
train.isnull().sum()
train = train[['msno', 'song_id', 'source_system_tab', 'source_type', 'target']]
train

train['source_system_tab'].unique()
train['source_system_tab'].value_counts()
train = train[~train['source_system_tab'].isin(['notification', 'settings'])]
train


# In[239]:


songs = pd.read_csv('/Users/lillypeters/Downloads/songs.csv')


# In[240]:


members = pd.read_csv('/Users/lillypeters/Downloads/members.csv')
members = members[['bd', 'msno', 'city']]
members.dropna()


# In[241]:


song_extra = pd.read_csv('/Users/lillypeters/Downloads/song_extra_info.csv')
song_extra


# In[255]:


merged_song = pd.merge(songs, song_extra[['song_id', 'name']], on='song_id', how='left')


# In[243]:


merged_song.dropna()


# In[244]:


merged_song.dropna()


# In[245]:


merged_song = merged_song[['artist_name', 'name', 'song_id']]
merged_song


# In[256]:


merged_song[merged_song.duplicated()]
print(merged_song.duplicated().sum())


# In[257]:


merged_song = merged_song.drop_duplicates()


# In[258]:


colab_df= pd.merge(train, merged_song[['song_id', 'name', 'artist_name']], on='song_id', how='left')


# In[259]:


colab_df


# In[260]:


kaggle = pd.merge(colab_df, members[['bd', 'msno', 'city']], on='msno', how='left')


# In[261]:


kaggle.rename(columns={'bd': 'age'})


# In[262]:


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


# In[265]:


kaggle['source_system_tab'].unique()


# In[264]:


kaggle['source_system_tab'].unique()
kaggle['source_system_tab'].value_counts()

