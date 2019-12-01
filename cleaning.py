# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:33:30 2019

@author: Gad
"""

import pandas as pd
import re 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from lookup import contractions
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

tokenizer = TweetTokenizer()
lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.add("wikipedia")




train_data = pd.read_csv("train.csv")

def cleaning():    
        #convert to lowercase
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        str(string).lower())
              
        #remove newline special character
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('\n',' ',str(string)))
        
        #remove punctiations
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','',str(string)))
                     
        #remove ipdadress
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        re.sub('\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}','',str(string)))
            
        #remove urls
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub("(http://.*?\s)|(http://.*)|(https://.*?\s)|(https://.*)",
                     '',str(string)))
            
        #remove text beginging with user
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub('\[\[User.*','',str(string)))
        
        #remove numbers
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
            re.sub('[0-9]','',str(string)))
            
        #stemming and expansion
    train_data["comment_text"] = train_data["comment_text"].map(lambda string:
        preprocess(string))
            
            
    return train_data

def preprocess(comment):
    words = tokenizer.tokenize(comment)
    words = [contractions[word] if word in contractions else word for word in words]
    words = [lemm.lemmatize(word, "v") for word in words]
    words = [x for x in words if not x in stop_words]
    
    
    processed =" ".join(words)
    return processed
       
     
def neutral_label(row):
    if(row.toxic + row.severe_toxic + row.obscene + row.threat + row.insult +
                   row.identity_hate > 0):
        return 1
    else:
        return 0
    



def main():
    train_data = cleaning()
    neutral = train_data.apply(lambda row: neutral_label(row),axis = 1)
    train_data = train_data.drop(columns = ['toxic', 'severe_toxic', 'obscene', 
                           'threat', 'insult', 'identity_hate'])
    
    train_data['toxic'] = neutral
         
    train_data.to_csv('clean_train.csv', index = False, float_format='%.3f')
    return train_data
    
    
if __name__ == "__main__":
    train = main()
    
    
    
    
            

            
            
            
        
        
        
                     
        
                     
       
        
        
    
    




