# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:05:14 2019

Name:Gad Kibet & Harrison Govan
Project: CS701 Senior seminar
"""
#Import libraries 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import  GlobalMaxPool1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score,accuracy_score



COMMENT = "comment_text"
MAX_FEATURES = 20000 #maximum number of training features 
MAX_SEQUENCE_LENGTH = 200 #Maximum length of taining input
batch_size = 100 #set batch size

"""Load data"""
train_data = pd.read_csv("clean_train.csv")
train_data.comment_text = train_data.comment_text.astype(str)

"""Get comments and labels"""
y_values = train_data["toxic"].values
x_values_train = train_data[COMMENT]


"""Tokenize and convert texts to integer sequences"""
tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(x_values_train))

tokenized_train = tokenizer.texts_to_sequences(x_values_train)

"""Pad/clip input"""
x_train = pad_sequences(tokenized_train,maxlen = MAX_SEQUENCE_LENGTH)

"""Split data to training and test data"""
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_values,test_size=0.2,
               random_state=13)

"""Define model architecture"""
def create_model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    embed_dim = 128
    embed_layer = Embedding(MAX_FEATURES,embed_dim)(input_layer)
    lstm_layer = LSTM(60, return_sequences=True,name='lstm_layer')(embed_layer)
    pooling_layer = GlobalMaxPool1D()(lstm_layer)
    dropout = Dropout(0.1)(pooling_layer)
    dense_layer = Dense(50, activation="relu")(dropout)
    dropout_2 = Dropout(0.1)(dense_layer)
    output = Dense(1, activation="sigmoid")(dropout_2)
    
    model = Model(inputs = input_layer, outputs = output)
    model.compile(optimizer ='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model

"""Fit and evaluate model"""
def fit_model():
    epochs = 1
    model = create_model()
    model.fit(x_train,y_values, batch_size=batch_size, 
                        epochs=epochs, validation_split=0.1)
    y_pred = model.predict(X_test,verbose=0)
    
    #Print evaluation statistics
    print("ROC_SCORE: {}".format(roc_auc_score(Y_test,y_pred)))
    y_pred = (y_pred > 0.5).astype(int)
    print("confusion matrix:")
    accuracy = accuracy_score(Y_test,y_pred)
    print("The accuracy is: {}".format(accuracy))
    print(confusion_matrix(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))
    
    

if __name__ == "__main__":
    fit_model()
    
    
    
    
    








