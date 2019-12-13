# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:56:14 2019

@author: Gad & Gad
"""

import pandas as pd
from matplotlib import pyplot as plt;plt.rcdefaults()
import numpy as np

ref = pd.read_csv('train.csv')
df = pd.read_csv('clean_train.csv')
df = df.dropna()

"""Find comment length"""
def commentL(comment):
    comment = comment.split()
    return len(comment)

#create new commment length column for visualization
df["charLength"] = df["comment_text"].apply(lambda comment:commentL(comment))
toxic = df[(df['toxic']>0)]
non_toxic = df[(df['toxic']<1)]


"""Create various visualizations"""
def main():
    print("\n")
    print("Summary Statistics")
    

    
    #Data Distribution
    classes = ["Toxic","Non-toxic"]
    counts = [len(toxic),len(non_toxic)]
    colors = ["red",'coral']

    #plot pie chart
    explode = (0.1,0)
    fig1, ax1 = plt.subplots()
    pies = ax1.pie(counts, explode=explode,
            shadow=True, startangle=90, colors = colors,
            autopct='%1.1f%%')
    ax1.axis('equal') 
    plt.title("Data Distribution", weight = "bold", size = 14)
    plt.legend(pies[0],classes, fontsize = 10, shadow = True)
    plt.tight_layout()
    plt.savefig('Visualizations\dataDistirbution.png',dpi = 300, facecolor='white')
    plt.show()
    
   

    

    #Comment length visualization
    plt.hist(df["charLength"].values,bins =100, range=(0,400))
    plt.title("Comment Length Distribution", weight="bold", size = 14)
    plt.xlabel("Comment Length",fontweight="bold")
    plt.ylabel("Number of observations",fontweight="bold")
    plt.tight_layout()
    plt.savefig('Visualizations\commentLength.png',dpi = 300, facecolor='white')
    plt.show()
    
    
    #display F-1 Score visualization for results
    results = [0.67,0.78,0.80,0.82]
    algs = ("NB","LR","CNN","RNN")
    real = ("Naive Bayes","Logistic Regression","Convolutional Neural Network",
              "Recurrent Neural Network")
    y_pos = np.arange(len(algs))
    
    color = ['#6c5b7b','#c06c84','#f67280','#f8b195']
    
    fig1, ax1 = plt.subplots()
    tick_label = ax1.bar(y_pos, results, align='center',color = color)
    plt.xticks(y_pos, algs)
    plt.legend(tick_label,real,bbox_to_anchor=(1,1), loc='best',
               frameon=True,fontsize='x-small',shadow=True)
    autolabel(tick_label,ax1)
    plt.xlabel("Algorithms",fontweight='bold')
    plt.ylabel('F1-score',fontweight='bold')
    plt.title('Results (F1-score)', weight="bold", size = 14)
    plt.tight_layout()
    plt.savefig('Visualizations\score.png',dpi = 300, facecolor='white')
    plt.show()
    
    
    #Display accuracy
    results = [0.95,0.96,0.97,0.98]
    algs = ("NB","LR","CNN","RNN")
    real = ("Naive Bayes","Logistic Regression","Convolutional Neural Network",
              "Recurrent Neural Network")
    y_pos = np.arange(len(algs))
    
    color = ['#6c5b7b','#c06c84','#f67280','#f8b195']
    
    fig1, ax1 = plt.subplots()
    tick_label = ax1.bar(y_pos, results, align='center',color = color)
    plt.xticks(y_pos, algs)
    plt.legend(tick_label,real,bbox_to_anchor=(1,1), loc='best',
               frameon=True,fontsize='x-small',shadow=True)
    autolabel(tick_label,ax1)
    plt.xlabel("Algorithms",fontweight='bold')
    plt.ylabel('Accuracy',fontweight='bold')
    plt.title('Results (Accuracy)', weight="bold", size = 14)
    plt.tight_layout()
    plt.savefig('Visualizations\scoreAccuracy.png',dpi = 300, facecolor='white')
    plt.show()
    
    
    # set width of bar
    barWidth = 0.25
 
    # set height of bar
    recall = [0.53, 0.68, 0.76, 0.73]
    precision= [0.91, 0.90, 0.84, 0.92]
 
    # Set position of bar on X axis
    r1 = np.arange(len(recall))
    r2 = [x + barWidth for x in r1]

    fig1, ax1 = plt.subplots()
    # Make the plot
    rect_1 = ax1.bar(r1, recall, color='#6c5b7b', width=barWidth, edgecolor='white')
    rect_2 = ax1.bar(r2, precision, color='#f8b195', width=barWidth, edgecolor='white')

    autolabel(rect_1,ax1)
    autolabel(rect_2,ax1)
    # Add xticks on the middle of the group bars
    plt.title("Recall and Precison Scores",fontweight='bold',size = 14)
    plt.ylabel("Reacall and Precision scores", fontweight='bold')
    plt.xlabel('Algorithms', fontweight='bold')
    locs, labels = plt.xticks([r + barWidth for r in range(len(recall))], ["NB", "LR", "CNN", "RNN"])
 
    # Create legend & Show graphic
    plt.legend(("Recall","Precision"),bbox_to_anchor=(1,1), loc='best',
               frameon=True,fontsize='x-small',shadow=True)


    plt.tight_layout()

    plt.savefig('Visualizations\modelPrec.png',dpi = 800, facecolor='white')
    plt.show()

    
    

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        

if __name__ == "__main__":
    main()


