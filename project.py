# -*- coding: utf-8 -*-
"""
Name:Gad Kibet & Harrison Govan
Project: CS701 Senior seminar
"""

import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('train.csv')
labels = ['obscene','insult','toxic',
              'severe_toxic','identity_hate','threat']
    



def train():
    x_vals = train_data["comment_text"]
    
    """
    dataFrame.iloc[rows,columns] since our y values stretch from the second
    to the eigth column
    
    """
    y_vals = train_data.iloc[:,2:8]
    
    """
    Split the training data into training set and test set in 80:20 
    ratio. Most recommended split
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
                                                        x_vals, 
                                                        y_vals, 
                                                        test_size= 0.2,
                                                        random_state=13
                                                        )
    
    #Vectorization (i.e. create a matrix with term
    #frequency invert document frequency)
    word_vectorizer = TfidfVectorizer(
                             ngram_range=(1,2),
                             tokenizer=TweetTokenizer().tokenize,
                             min_df=3, 
                             max_df=0.9,
                             strip_accents='unicode', 
                             use_idf=1,
                             smooth_idf=1, 
                             sublinear_tf=1,

                             max_features = 20000
                             )
        
    """
    We perform fit transform on the training data so that we can learn 
    both the parameters of scaling on the train and scale the data. On the test
    data we only tranform the data so that we use the scaling used in the train
    data
    """
    training_vect = word_vectorizer.fit_transform(X_train)
    test_vect = word_vectorizer.transform(X_test)
    
    
    #clasify perfroming logistic refression
    losses = []
    auc = []
    for class_name in labels:
        #run logistic regression on each class 
        print("Fit :{}".format(class_name))
        train_target = Y_train[class_name]
        test_target = Y_test[class_name]
        classifier = LogisticRegression(solver='sag',C=13)
        
        """
        compute cross validation score using standard 10-fold CV
        """
        cv_loss = np.mean(
                        cross_val_score(
                                        classifier, 
                                        training_vect, 
                                        train_target,
                                        cv=10, 
                                        scoring='neg_log_loss'
                                        )
                        )
        losses.append(cv_loss)
        print('CV Log_loss score for class {} is {}'.format(class_name, cv_loss))
        
        
        cv_score = np.mean(
                        cross_val_score(
                                        classifier, 
                                        training_vect, 
                                        train_target,
                                        cv=10, 
                                        scoring='accuracy'
                                        )
                        )
                        
        print('CV Accuracy score for class {} is {}'.format(class_name, cv_score))
        
        
        """
        Fit the model according to the training data
        """
        classifier.fit(training_vect, train_target)
        
        """
        Predict class labels for our test sample
        """
        y_pred = classifier.predict(test_vect)
        
        #Return probability estimates 
        y_pred_prob = classifier.predict_proba(test_vect)[:, 1]
        
        """
        Compute the area under the curve of true positive rate vs false positive
        rate. The higher the value the more accurate the classification is.This
        means that we have a very high true positive rate and a low false
        positive rate
        """
        auc_score = metrics.roc_auc_score(test_target, y_pred_prob)
        auc.append(auc_score)
        print("CV ROC_AUC score {}\n".format(auc_score))
        
        print(confusion_matrix(test_target, y_pred))
        print("\n")
        #generates a text report of the classification metrics detailing other
        #metrics such as precision and recall
        print(classification_report(test_target, y_pred))
        
    print('Total average CV Log_loss score is {}'.format(np.mean(losses)))
    print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))      
    

if __name__ == "__main__":
    train()








