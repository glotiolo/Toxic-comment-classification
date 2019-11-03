#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')



def train():
    df_label = train_data.iloc[:, 2:8]

    X_train, X_test, y_train, y_test = train_test_split(
        train_data["comment_text"], df_label, test_size=0.2, random_state=109)

    # Tokenization with NLTK and training
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        tokenizer=TweetTokenizer().tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        max_features=10000
    )

    # Create the bag of words using our vectorizer
    training_vect = word_vectorizer.fit_transform(X_train)

    test_vect = word_vectorizer.transform(X_test)

    return training_vect, test_vect, y_train, y_test, word_vectorizer


def main():
    classes = ["obscene","insult","toxic","severe_toxic","identity_hate","threat"]
    x_train, x_test, y_train, y_test = train()
    print("done1")

    #Create a Gaussian Classifier
    gnb = MultinomialNB()

    #Train the model using the training sets
    for label in classes:
        print("Fitting {}".format(label))
        train_target = y_train[label]
        test_target = y_test[label]

        gnb.fit(x_train, train_target)
        #Predict the response for test dataset
        y_pred = gnb.predict(x_test)

        # Model Accuracy, how often is the classifier correct?
        acc = gnb.score(x_test,test_target)
        print("{} accuracy is {}".format(label,acc))
        print(confusion_matrix(test_target, y_pred))
        print(classification_report(test_target, y_pred))



if __name__ == "__main__" :
    main()