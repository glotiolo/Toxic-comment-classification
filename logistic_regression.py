# -*- coding: utf-8 -*-
"""
Name:Gad Kibet & Harrison Govan
Project: CS701 Senior seminar
"""
#import libraries 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time




#Load data
train_data = pd.read_csv('clean_train.csv')
train_data.comment_text = train_data.comment_text.astype(str)

COMMENT = "comment_text"
x_vals = train_data[COMMENT]
y_vals = train_data["toxic"]

#Vectorization (i.e. create a matrix with term
#frequency invert document frequency)
word_vectorizer = TfidfVectorizer(
                             ngram_range=(1,1),
                             strip_accents='unicode', 
                             use_idf=True,
                             smooth_idf=True, 
                             sublinear_tf=False,
                             max_features = 20000,
                             )
    
"""
Split the training data into training set and test set in 80:20 
ratio. Most recommended split
"""
X_train, X_test, Y_train, Y_test = train_test_split(x_vals, y_vals, 
                                                    test_size= 0.2,
                                                    random_state=13)
                            
"""
We perform fit transform on the training data so that we can learn 
both the parameters of scaling on the train and scale the data. On the test
data we only tranform the data so that we use the scaling used in the train
data
"""
training_vect = word_vectorizer.fit_transform(X_train)
        
test_vect = word_vectorizer.transform(X_test)
        

#get model and fit
def get_model(x,y):
    model = LogisticRegression(solver='liblinear',penalty = 'l1')
    return model.fit(x,y)


""" Evaluate"""
def main():
    start_time = time()
    classifier = get_model(training_vect,Y_train)
    y_pred = classifier.predict(test_vect)
    y_pred_prob = classifier.predict_proba(test_vect)[:,1]
    """
        Compute the area under the curve of true positive rate vs false positive
        rate. The higher the value the more accurate the classification is.This
        means that we have a very high true positive rate and a low false
        positive rate
     """
    auc_score = metrics.roc_auc_score(Y_test, y_pred_prob)
    accuracy = metrics.accuracy_score(Y_test,y_pred)
    print("The accuracy is: {}".format(accuracy))
    print("CV ROC_AUC score {}\n".format(auc_score))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test,y_pred))
    print("The duration was: {} seconds".format(round(time()-start_time,2)))
    
    
    
     
if __name__ == "__main__":
    main()
    
    
    







