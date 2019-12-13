# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:33:30 2019

@author: Gad & Harrison
"""
#Import libraries 
import pandas as pd
import re 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from lookup import contractions
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

#Define tokenizer, lemmatizer and generate set of stopwords
tokenizer = TweetTokenizer()
lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.add("wikipedia")





#Load data
train_data = pd.read_csv("train.csv")

"""Clean data"""
def cleaning():    
        #convert to lowercase
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        string.lower())
              
        #remove newline special character
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('\n',' ',string))
        
        #remove punctiations
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','',string))
                     
        #remove ipdadress
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}','',string))
            
        #remove urls
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub("(http://.*?\s)|(http://.*)|(https://.*?\s)|(https://.*)",
                     '',string))
            
        #remove text beginging with user
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub('\[\[User.*','',string))
        
        #remove numbers
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub('[0-9]','',string))
            
        #stemming and expansion
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        preprocess(string))
        
        #remove special characters
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub('[^A-Za-z0-9]+', ' ', string))
            
            
    return train_data


"""Lemmatize,expand and tokenize the data"""
def preprocess(comment):
    words = tokenizer.tokenize(comment)
    words = [contractions[word] if word in contractions else word for word in words]
    words = [lemm.lemmatize(word, "v") for word in words]
    words = [x for x in words if not x in stop_words]
    
    processed =" ".join(words)
    return processed
       
"""Consolidate data into two classes"""   
def neutral_label(row):
    #Assign 1 label if belongs to any of these classes
    if(row.toxic + row.severe_toxic + row.obscene + row.threat + row.insult +
                   row.identity_hate > 0):
        return 1
    else:
        return 0
    


"""Consolidate data and save as excel file"""

def main():
    train_data = cleaning()
    neutral = train_data.apply(lambda row: neutral_label(row),axis = 1)
    train_data = train_data.drop(columns = ['toxic', 'severe_toxic', 'obscene', 
                           'threat', 'insult', 'identity_hate'])
    
    train_data['toxic'] = neutral
         
    train_data.to_csv('clean_train.csv', index = False, float_format='%.3f')

    
    
if __name__ == "__main__":
    main()

    
    
            

            
            
            
        
        
        
                     
        
                     
       
        
        
    
    




