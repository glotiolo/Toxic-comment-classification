# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:34:09 2019

@author: Gad
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten
from keras.layers import Embedding
from keras.callbacks import EarlyStopping 
from keras.layers.convolutional import Conv1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score




train_data = pd.read_csv("clean_train.csv")
train_data.comment_text = train_data.comment_text.astype(str)




batch_size = 100
COMMENT = "comment_text"
MAX_FEATURES = 20000 #maximum number of features
MAX_SEQUENCE_LENGTH = 200 #maximum length of comments


""" Get the comments and labels"""
y_values = train_data['toxic'].values
x_values_train = train_data[COMMENT]



""" Tokenize comments"""
tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(x_values_train))

tokenized_train = tokenizer.texts_to_sequences(x_values_train)


"""Pad comments to make them even length"""
x_train = pad_sequences(tokenized_train,maxlen = MAX_SEQUENCE_LENGTH)


"""Split training data"""
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_values,test_size=0.2,
               random_state=13)



"""Define the CNN model"""
def create_Model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    embed_dim = 128
    embed_layer = Embedding(MAX_FEATURES,embed_dim)(input_layer)
    conv_1 = Conv1D(filters=128, kernel_size=8, 
                    padding='same', activation='relu')(embed_layer)
    pool_1 = MaxPooling1D(5)(conv_1)
    conv_2 = Conv1D(filters=128, kernel_size=8, 
                    padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling1D(5)(conv_2)
    conv_3 = Conv1D(filters=128, kernel_size=8, 
                    padding='same', activation='relu')(pool_2)
    pool_3 = MaxPooling1D(5)(conv_3)
    flatten = Flatten()(pool_3)
    dense = Dense(128, activation='relu')(flatten)
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs = input_layer, outputs = output)
    model.compile(optimizer ='adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model


"""Fitting model"""
def fit_model():
    epochs = 1
    model = create_Model()
    model.summary()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            min_delta=0.0001)])
    
    y_pred = model.predict(X_test,verbose=0)
    print("ROC_SCORE: {}".format(roc_auc_score(Y_test,y_pred)))
    y_pred = (y_pred > 0.5).astype(int)
    print("confusion matrix:")
    print(confusion_matrix(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))
    return history,y_pred



if __name__ == "__main__":
    history,y_hat = fit_model()   

     
    


    














    

    
