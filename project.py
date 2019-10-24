# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
# import numpy as np
from nltk.tokenize import TweetTokenizer
# from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xlwings as xw
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#subm = pd.read_csv('sample_submission.csv')
labels = ['obscene','insult','toxic',
              'severe_toxic','identity_hate','threat']
    
    


def train():
    df_label = train_data.iloc[:,2:8]

    X_train, X_test, y_train, y_test = train_test_split(
        train_data["comment_text"], df_label, test_size=0.2, random_state=109)

    #Tokenization with NLTK and training
    word_vectorizer = TfidfVectorizer(
                             ngram_range=(1,2),
                             tokenizer=TweetTokenizer().tokenize,
                             min_df=3, 
                             max_df=0.9,
                             strip_accents='unicode', 
                             use_idf=1,
                             smooth_idf=1, 
                             sublinear_tf=1,
                             max_features=10000
                             )
    
    #Create the bag of words using our vectorizer
    training_vect = word_vectorizer.fit_transform(X_train)

    test_vect = word_vectorizer.transform(X_test)



    return training_vect, test_vect, y_train, y_test, word_vectorizer



if __name__ == "__main__":
    training_vect, test_vect, y_train, y_test, word_vectorizer = train()
    print(word_vectorizer.vocabulary)









