# -*- coding: utf-8 -*-
"""
PIC 16B Final Project functions
"""

def decode_songs(song_indices, song_encoder):
    """
    Function that decodes the label encoded songs

    ARGS:
    song_indecies: the label encoded songs
    spng_encoder: label encoding 

    RETURNS:
    the decoded songs
    """
    song_indices = np.atleast_1d(song_indices)
    
    return song_encoder.inverse_transform(song_indices)


def at_k(W, H, train_matrix, test_lookup, k=10):
    """
    https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k
    Function to return the Precision@K which is how relevant the top K recommended songs are and Recall@K which is how many of the 
    recommended songs are relevant are in the top K  

    ARGS:
    W: user latent features
    H: song latent features
    train_matrix: sparse matrix
    test_lookup: dictionary of grouped testing data to compare for relevance
    k: top K songs that we compare

    RETURNS:
    Precision@K, Recall@K value
    """

    precisions = []
    recall = []

    # looks at the relevance for each user then averages it out
    for user_idx in test_lookup.keys():

        # the top k recommendations for each user
        recommended = recommend_user(user_idx,W,H,train_matrix,top_k=k)
        
        # all the songs that a user listened to
        listened = test_lookup[user_idx]

        # relevance is def as overlap between how many are listened to from the top k recommendations
        relevant = len(set(recommended) & listened)
        
        # precision = number of relevant items in K / total number of items in K
        precisions.append(relevant / k)

        # recall = number of relevant items in K / total number of relevant items
        if len(listened) > 0:
            recall.append(relevant / len(listened))

    return np.mean(precisions), np.mean(recall)


def user_at_k(user_idx, W, H, train_matrix, test_lookup, k=10):

    recommended = recommend_user(user_idx,W,H,train_matrix,top_k=k)

    listened = test_lookup.get(user_idx, set())

    relevant = len(set(recommended) & listened)

    precision = relevant / k

    if len(listened) > 0:
        recall = relevant / len(listened)
    else:
        recall = 0

    return precision, recall


def recommend_user(user_idx, W, H, train_matrix, top_k=10):
    """
    Function to return K top recommended songs based on a user

    ARGS:
    user_idx: the user to recommend to
    W: user latent features
    H: song latent features
    train_matrix: the sparse matrix
    top_K: integer value of TOP K songs to return (default 10)

    RETURNS:
    top_song_indices: sorted songs by top K recommended
    """

    # predict scores based on user id
    scores = W[user_idx] @ H

    # REMOVE already seen songs
    seen_songs = train_matrix[user_idx].indices
    scores[seen_songs] = -np.inf

    # sort songs by most recommended
    top_song_indices = np.argsort(scores)[::-1][:top_k]

    return top_song_indices


def train_test_split(df, test_size=0.4):
    """
    Function to randomly shuffle our data frame by each user  
    Split it into training and testing dataframes

    ARGS:
        df - user dataframe
        test_size - default 0.2. 20% of each user into testing 

    RETURNS: split dataframes
        train_df: the training dataframe (1-test_size) %
        test_df: testing dataframe  (test_size) %
    """

    train_rows = []
    test_rows = []

    # splitting based on user so that there is TEST_SIZE % in testing of each user
    for user, user_df in df.groupby('user_idx'):

        # shuffle user df so that we can have random train/testing 
        user_df = user_df.sample(frac=1, random_state=42)

        n_test = max(1, int(len(user_df) * test_size))
        test = user_df.iloc[:n_test]
        train = user_df.iloc[n_test:]

        train_rows.append(train)
        test_rows.append(test)

    # make into pandas df
    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    return train_df, test_df



def build_knn_graph(song_embeddings, n_neighbors=6, metric="cosine"):
    """
    Function that builds KNN graph

    ARGS:
        song_embeddings
        n_neighbors: default 6 
        metric: default cosine

    RETURNS: 
        G
        KNN
    """
    # Build KNN graph from CNN embeddings
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(song_embeddings)

    distances, indices = knn.kneighbors(song_embeddings)

    G = nx.Graph()

    for i in range(len(song_embeddings)):
        G.add_node(i)

        # j=0 is the song itself, so skip it
        for j in range(1, n_neighbors):
            neighbor = indices[i][j]
            distance = distances[i][j]

            G.add_edge(i, neighbor, weight=distance)

    return G, knn

def recommend(song_name, song_data=df.copy(), G=G):
    """
    Function that builds KNN graph

    ARGS:
        song_name: name of song
        song_data: dataframe
        G: G

    RETURNS: 
        top 5 recommended songs based on song name
    """
    source_idx = find_song_index(song_data, song_name)

    direct_neighbors = set(G.neighbors(source_idx))

    path_length = nx.single_source_dijkstra_path_length(
        G,
        source_idx,
        weight="weight"
    )

    candidates = [
        (idx, dist)
        for idx, dist in path_length.items()
        if idx != source_idx and idx not in direct_neighbors
    ]# get the songs based on shortest path distance in the KNN graph, excluding direct neighbors and the song itself

    top_5 = sorted(candidates, key=lambda x: x[1])[:5] # sort candidates by distance and take top 5

    rec_indices = [idx for idx, dist in top_5]
    scores = [dist for idx, dist in top_5]

    recs = song_data.loc[
        rec_indices,
        ["track_name", "artists", "track_genre"]
    ].copy()

    recs["distance"] = scores
    recs["similarity"] = 1 / (1 + recs["distance"])

    return recs

def recommend_cnn_knn(song_name, song_data=df.copy(), song_embeddings=song_embeddings, knn=knn_cnn, top_k=5):
    """
    Function that recommend songs using KNN on CNN embeddings

    ARGS:
        song_name: name of song
        song_data: dataframe
        G: G
        song_embeddings: song_embeddings
        knn: knn_cnn
        top_k: default 5, top k songs to return
    RETURNS: 
        top 5 recommended songs based on song name
    """
    source_idx = find_song_index(song_data, song_name)

    distances, indices = knn.kneighbors(
        song_embeddings[source_idx].reshape(1, -1),
        n_neighbors=6
    )

    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]

    recs = song_data.iloc[rec_indices][
        ["track_name", "artists", "track_genre"]
    ].copy()

    recs["distance"] = rec_distances
    recs["similarity"] = 1 / (1 + recs["distance"])

    return recs
