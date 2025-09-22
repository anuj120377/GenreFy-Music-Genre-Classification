# GenreFy-Music-Genre-Classification

<img width="717" height="290" alt="image" src="https://github.com/user-attachments/assets/af45b8c6-6d48-41c5-a424-9869e36fdee5" />


Music Genre Classification using Machine Learning

This project focuses on predicting the genre of music based on audio features using multiple machine learning algorithms. The dataset includes features extracted from music tracks, and the goal is to classify each track into its correct genre.

### **Motivation -**

So, music and audio are basically sequential data, and a lot of people immediately jump to using deep learning LSTMs, GRUs, or even CNNs to classify genres. But I thought: why make it so complicated? Deep models take a lot of resources, can be tricky to tune, and honestly, sometimes you don’t even need them if you do some good feature engineering.

I wanted to see if I could get solid accuracy using classical ML models, just by working smart with the features.

### Dataset
The dataset used is GTZAN (the famous GTZAN dataset, the MNIST of sounds)

The GTZAN dataset contains 1000 audio files. Contains a total of 10 genres, each genre contains 100 audio files

1.Blues

2.Classical

3.Country

4.Disco

5.Hip-hop

6.Jazz

7.Metal

8.Pop

9.Reggae

10.Rock

Genres original
A compilation of ten genres, each with 100 audio recordings, each lasting 30 seconds (the famous GTZAN dataset, the MNIST of sounds)

Images original
Each audio file has a visual representation. Neural networks are one technique to classify data because they usually take in some form of picture representation.

CSV files
The audio files' features are contained within. Each song lasts for 30 seconds long has a mean and variance computed across several features taken from an audio file in one file. The songs are separated into 3 second audio files in the other file, which has the same format.


###  **My Approach -**

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

### **Performance -**

| Model                | Accuracy (%) |
| -------------------- | ------------ |
| KNN                  | 88.02        |
| SVM                  | 85.21        |
| Logistic Regression  | 72.44        |
| Gaussian Naive Bayes | 52.56        |
| Voting Ensemble      | 85.02        |

Observation: KNN actually turned out to be the best single model, and the ensemble was close but didn’t beat it.


### **Conclusion -**

So, after trying out different models, I found that KNN worked the best with an accuracy of 88%, while the ensemble model with all four classifiers got around 85%. This shows that sometimes the best single model can beat a combination of models!
The main takeaway is that you don’t always need deep learning to get good results. By understanding the data and extracting meaningful features ML models can still perform really well.


### **Improvements / Future Work -**

-> Hyperparameter tuning for KNN, SVM, and the ensemble to squeeze out better accuracy.

-> Experiment with stacking or boosting ensembles for potentially improved performance.

-> Use feature selection or dimensionality reduction to make models faster, cleaner, and potentially more accurate.

-> Perform more advanced feature extraction, including additional spectral, temporal, and rhythm-based features to better capture music characteristics.

-> Test on larger and more diverse datasets to ensure the models generalize well.

