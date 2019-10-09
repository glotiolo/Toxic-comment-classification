# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#subm = pd.read_csv('sample_submission.csv')
labels = ['obscene','insult','toxic',
              'severe_toxic','identity_hate','threat']
    
    


def train():
    #Tokenization with NLTK and training
    word_vectorizer = TfidfVectorizer(
                             ngram_range=(1,2),
                             tokenizer=TweetTokenizer().tokenize,
                             min_df=3, 
                             max_df=0.9,
                             strip_accents='unicode', 
                             use_idf=1,
                             smooth_idf=1, 
                             sublinear_tf=1 
                             )
    
    #Create the bag of words using our vectorizer
    training_vect = word_vectorizer.fit_transform(train_data["comment_text"])
    #test_vect = word_vectorizer.fit_transform(test_data["comment_text"])
    
    return training_vect, word_vectorizer
    
    
    
    
    


#
#if __name__ == "__main__":
#    train()








