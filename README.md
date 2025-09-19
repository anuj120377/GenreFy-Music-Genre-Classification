# GenreFy-Music-Genre-Classification

Music Genre Classification using Machine Learning

This project focuses on predicting the genre of music based on audio features using multiple machine learning algorithms. The dataset includes features extracted from music tracks, and the goal is to classify each track into its correct genre.

# Motivation

So, music and audio are basically sequential data, and a lot of people immediately jump to using deep learning—LSTMs, GRUs, or even CNNs—to classify genres. But I thought: why make it so complicated? Deep models take a lot of resources, can be tricky to tune, and honestly, sometimes you don’t even need them if you do some good feature engineering.

I wanted to see if I could get solid accuracy using classical ML models, just by working smart with the features.

# My Approach
Feature Extraction: I focused on using extracted handcrafted audio features like:
MFCCs (Mel-frequency cepstral coefficients)
Chroma features
Spectral contrast
Tonnetz features
Models Tried:

K-Nearest Neighbors (KNN) → Best accuracy 88.02%
Support Vector Machine (SVM)
Multinomial Logistic Regression
Gaussian Naive Bayes (GNB)
Ensemble: Combined all four using a Voting Classifier (soft voting) to see if they could do better together.

Evaluation: Used cross-validation to make sure the results weren’t just random luck.
