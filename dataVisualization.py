# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:56:14 2019

@author: Gad
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
labels = ['obscene','insult','toxic',
              'severe_toxic','identity_hate','threat']




def main():
    print("\n")
    print("Summary Statistics")
    
    #Find length of each comment
    lenSeries = df.comment_text.str.len()
    df["charLength"] = lenSeries
    
    
    
    
    print("Training set size: {}".format(len(df)))
    print("-" * 38)
    
    #check for null comments
    no_comment = df[df['comment_text'].isnull()]
    print("Comment Statistics")
    print("mean length: {}".format(lenSeries.mean()))
    print("Standard Deviation: {}".format(lenSeries.std()))
    print("Longest comment: {}".format(lenSeries.max()))
    print("The number of null comments is: {}".format(len(no_comment)))
    lenSeries.hist()
    plt.title("Comment Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()

    
    print("-" * 38)
    print("Summary for each label")
    print(df[labels].sum())
    print("-" * 38)
    
    
    #Find the the number of rows with these labels
    toxic = df['toxic'].sum()
    severe_toxic = df['severe_toxic'].sum()
    obscene = df['obscene'].sum()
    threat = df['threat'].sum()
    insult = df['insult'].sum()
    identity_hate = df['identity_hate'].sum()
    
    #Find the number of unlabelled comments i.e don't belong to any category
    neutral = len(df[(df['toxic']!=1) & (df['severe_toxic']!=1) & 
                 (df['obscene']!=1) & (df['threat']!=1) & 
                 (df['insult']!=1) & (df['identity_hate']!=1)])
    
    #Find the number of datapoints that intersect
    overlap = len(df[(df['toxic']!=0) & (df['severe_toxic']!=0) & 
                 (df['obscene']!=0) & (df['threat']!=0) & 
                 (df['insult']!=0) & (df['identity_hate']!=0)])
    
    
     #plot bargraph for visalizations      
    frequency = [toxic,severe_toxic,obscene,threat,insult,identity_hate,neutral,overlap]
   
    categories = ('toxic','severe_toxic','obscene','threat',
               'insult','identity_hate','neutral','overlap')
    
    x_pos = np.arange(len(categories))
    plt.barh(x_pos, frequency, align='center', alpha=1.0)
    plt.yticks(x_pos,categories)
    plt.xlabel("Class Distribution")
    plt.title("Data Distirbution")
    plt.show()
    
    #plot pie chart
    explode = (0.1,0,0,0,0,0,0,0)
    fig1, ax1 = plt.subplots()
    ax1.pie(frequency, explode=explode, labels=categories,
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Data Distribution")
    plt.show()
    
    
if __name__ == "__main__":
    main()


