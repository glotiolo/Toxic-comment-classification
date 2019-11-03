# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:33:30 2019

@author: Gad
"""

import pandas as pd
import re 
import nltk as nk
import numpy as np



train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
snapshot = train_data.copy()


def cleaning():
    for entry in train_data,test_data:
        #convert to lowercase
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            str(string).lower())
            
            
        #remove newline special character
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub('\n',' ',str(string)))
        
        #remove punctiations
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','',str(string)))
                     
        #remove ipdadress
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub('\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}','',str(string)))
            
        #remove urls
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub("(http://.*?\s)|(http://.*)|(https://.*?\s)|(https://.*)",
                     '',str(string)))
            
        #remove text beginging with user
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub('\[\[User.*','',str(string)))
        
        #remove numbers
        entry["comment_text"] = entry["comment_text"].map(lambda string:
            re.sub('[0-9]','',str(string)))
        
        
        
        

        
            
            
    train_data.to_csv(r'C:\Users\Gad\Documents\cs701\Senior Seminar\codebase\project\701-Seminar\clean_train.csv')
    test_data.to_csv(r'C:\Users\Gad\Documents\cs701\Senior Seminar\codebase\project\701-Seminar\clean_test.csv')
    
    
if __name__ == "__main__":
    cleaning()
    
    
    
            

            
            
            
        
        
        
                     
        
                     
       
        
        
    
    




