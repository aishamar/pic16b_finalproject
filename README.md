# pic16b_finalproject
PIC 16B Final Project 
pic16b_finalproject

PIC 16B Final Project

Spotify Recommendation System

Introduction

A music recommendation system is a system that uses various models to suggest songs to users. The goal of such a system is to personalize the music listening experience for each user, by providing them with a list of songs that they are likely to enjoy based on their listening history and preferences. The goal of recommendation systems is to provide personalized recommendations to users, improving their overall experience and increasing engagement.

Problem Definition

There are several challenges in developing a music recommendation system. We will start out by collecting data from Kaggle datasets. Then we will clean and analyze the data to find insights to better understand the relationship between songs or the interactions between songs and users. Later we will munge the data and apply normalization, encoding preprocessing. Once data is prepared, we aim to try two different branches: collaborative filtering and content filtering. We find models for each method which can recommend songs based on user and song data. We output recommendations to users and analyze the accuracy of the outputs.

Data Sources

Our main goal is to integrate this project with datasets: Kaggle and Kaggle.

Content-based Recommenders:

These systems recommend items that are similar to those a user has already liked in the past. The system analyzes the relationship between its features and genres of the items and suggests items with similar attributes.

Collaborative Filtering Recommenders:

These systems recommend items based on user to user interactions. The model will compare every users interactions with an item and find hidden patterns within those items to create groups.

Implementation:

Content-based Recommenders:

We will first KNN on our entire song dataset
We try CNN as our supplementary model
Once it is given a song, the system will compare its attributes and create latent factors
After creating latent factors, the system will then check distance between the input song and other songs. We will be utilizing ‘cosine similarity’ to find the distance for this project
The system will output top 5 recommendations
Now that we have designed our Recommender System. For the system, we mainly use KNN as our machine learning model. And we are trying to find different models to improve CNN, so we choose CNN. We create the model based on two different types of model, graph based and embedding based. To verify whether it does improve the KNN model, we also create a pure KNN model and create graphs to compare the output distances.

Collaborative Filtering Recommenders:

Build a sparse matrix based on whether a user repeated a song.
Use Non-negative matrix facorization to decompose our matrix into two lower ranked matrices and extract song features and weights for each song.
Each user will be given a song recommendation and have a certain amount of test songs they listened to, to determine if it's an accuracte recommendation based on recall@k and precision@k.
