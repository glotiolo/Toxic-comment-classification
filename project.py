# -*- coding: utf-8 -*-
"""
Name:Gad Kibet & Harrison Govan
Project: CS701 Senior seminar
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time


test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')
train_data = pd.read_csv('train.csv')
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


COMMENT = "comment_text"
x_vals = train_data[COMMENT]
   
#dataFrame.iloc[rows,columns] 
y_vals = train_data.iloc[:,2:]

"""
    Split the training data into training set and test set in 80:20 
    ratio. Most recommended split
"""
X_train, X_test, Y_train, Y_test = train_test_split(x_vals, y_vals, 
                            test_size= 0.2,random_state=13)
                            
test_x = test[COMMENT] #this is the test data we get from Kaggle

#Vectorization (i.e. create a matrix with term
#frequency invert document frequency)
word_vectorizer = TfidfVectorizer(
                             ngram_range=(1,1),
                             min_df=1, 
                             max_df=0.5,
                             strip_accents='unicode', 
                             use_idf=True,
                             smooth_idf=True, 
                             sublinear_tf=False,
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
        
real_test = word_vectorizer.transform(test_x)


#Fit a model for each dependent variable
def get_model(target,x):
    y = target.values
    model = LogisticRegression(solver='sag',C=13)
    return model.fit(x,y)


def train():
    start_time = time()

    auc = []
    
    predictions = np.zeros((len(test), len(labels)))
    
    for i,class_name in enumerate(labels):
        #run logistic regression on each class 
        print("Fit :{}".format(class_name))
        train_target = Y_train[class_name]
        test_target = Y_test[class_name] 
        
        classifier = get_model(train_target,training_vect)
        
        y_pred = classifier.predict(test_vect)
        y_pred_prob = classifier.predict_proba(test_vect)[:,1]
        
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
        print(classification_report(test_target, y_pred))
        
        #Return probability estimates for test submission
        predictions[:,i] = classifier.predict_proba(real_test)[:,1]
        
    print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))      
    print("The duration was: {} seconds".format(round(time()-start_time,2)))
    
    #Submission
    submid = pd.DataFrame({'id':sub['id']})
    submission = pd.concat([submid, pd.DataFrame(predictions,columns = labels)],
                                                axis = 1)
    
    submission.to_csv('submission.csv',index = False)

if __name__ == "__main__":
    train()
    
    
    








