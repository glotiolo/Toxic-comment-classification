# -*- coding: utf-8 -*-
"""
Name:Gad Kibet & Harrison Govan
Project: CS701 Senior seminar
"""
#Import Libraries 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('clean_train.csv')
train_data.comment_text = train_data.comment_text.astype(str)


"""Tokenize and vectorize data"""
def train():
    df_label = train_data["toxic"].values

    X_train, X_test, y_train, y_test = train_test_split(
        train_data["comment_text"], df_label, test_size=0.2, random_state=13)

    # Tokenization with NLTK and training
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        tokenizer=TweetTokenizer().tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        max_features=10000
    )

    # Create the bag of words using our vectorizer
    training_vect = word_vectorizer.fit_transform(X_train)
    
    #fit testing data
    test_vect = word_vectorizer.transform(X_test)

    return training_vect, test_vect, y_train, y_test


"""Fit and evaluate model"""
def main():
    x_train, x_test, y_train, y_test = train()

    #Create a Gaussian Classifier
    gnb = MultinomialNB()
    
    gnb.fit(x_train,y_train)
    #Predict the response for test dataset
    y_pred = gnb.predict(x_test)
    
    # Model Accuracy, how often is the classifier correct?
    acc = gnb.score(x_test,y_test)
    
    #Print evaluation statistics 
    print("Acuracy is {}".format(acc))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

if __name__ == "__main__" :
    main()