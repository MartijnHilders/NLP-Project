import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import collections
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
import math
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


def main():


    # importing the training data using pandas.
    df_train = pd.read_csv('Data/train.csv')

    # save the original text to easily inspect it and derive what has to be cleaned
    # df_train['comment_text'].to_csv('Data/OriginalText.csv')

    # save the cleaned text to easily inspect it
    # df_train['comment_text'].to_csv('Data/cleanedText.csv')

    # clean the text
    df_train['comment_text'] = df_train['comment_text'].apply(cleanText)


######################################################################################################################

    # test on small portion
    df_train = df_train.sample(n=25000, random_state=33)

    # setting up the X training comments (vectorize them to be able to be used as input for model) and Y training labels
    print("setting up training data ")
    Xtrain = word2Vec(df_train)
    Ytrain = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


    # importing the test data set to test the algorithm
    df_test = pd.read_csv('Data/test.csv')


    #run the algorithm for the first time and get an idea of the accuracy with the basic parameters.
    print("started training the model")
    rf_model = RandomForestClassifier()

    # for the whole dataset
    # rf_model.fit(Xtrain, Ytrain)

    # test the accuracy of the model on a split training dataset
    xtrain,xtest,ytrain,ytest = train_test_split(Xtrain,Ytrain,test_size=0.40, random_state=42)
    rf_model.fit(xtrain, ytrain)

    #prediction and crossvalidation
    rf_predict = rf_model.predict(xtest)
    rfc_cv_score = cross_val_score(rf_model, Xtrain, Ytrain, cv=10, scoring='roc_auc')

    #tests
    #TODO: explain the results.
    print("Model Accuracy")
    print("RF Accuracy: %0.2f%%" % (100 * rf_model.score(xtest, ytest)))


    print("Confusion Matrix:")
    print(confusion_matrix(ytest.values.argmax(axis=1), rf_predict.argmax(axis=1)))
    print('\n')


    print("Classification Report")
    print(classification_report(ytest, rf_predict))
    print('\n')


    print("All Cross Validation Scores")
    print(rfc_cv_score)
    print('\n')


    print("Mean Cross Validation Score")
    print(rfc_cv_score.mean())

################################################################################################################
    #
    # # improve the model by trying to find the best parameters for the random forest, we can do this by using RandomizedSearchedCV
    # # hyperparameter training, using 4 parameters
    # # only has to be run once!
    # # be careful for overfitting
    # # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    #
    # # number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    #
    # # number of features at every split
    # max_features = ['auto', 'sqrt']
    #
    # # max depth
    # max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
    # max_depth.append(None)
    #
    # # bootstrap
    # bootstrap = ['true', 'false']
    #
    # # create random grid
    # random_grid = {
    #     'n_estimators': n_estimators,
    #     'max_features': max_features,
    #     'max_depth': max_depth,
    #     'bootstrap': bootstrap
    # }
    #
    # # Random search of parameters
    # rfc_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid, n_iter=50, cv=2, verbose=2,
    #                                 random_state=42, n_jobs=-1)
    #
    # # Fit the model
    # rfc_random.fit(xtrain, ytrain)
    # # print results
    # print(rfc_random.best_params_)
    #
    # #result = {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 140, 'bootstrap': 'true'}

################################################################################################################

    # testing if the retrieved parameters are actually better
    # parameters from optimization step;
    # n_estimators = 200 , max_features = 'sqrt' , max_depth = 140 , bootstrap = 'true'

    #initializing the model and fitting it to the training data
    rf_optimized_model = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=140, bootstrap='true')
    rf_optimized_model.fit(xtrain,ytrain)

    # prediction and crossvalidation
    rf_predict_optimized = rf_optimized_model.predict(xtest)
    rfc_cv_score_optimized = cross_val_score(rf_optimized_model, Xtrain, Ytrain, cv=10, scoring='roc_auc')

    # tests
    # TODO: explain the results.
    print("Optimized Model Accuracy")
    print("RF Accuracy: %0.2f%%" % (100 * rf_optimized_model.score(xtest, ytest)))

    print("Optimized model Confusion Matrix:")
    print(confusion_matrix(ytest.values.argmax(axis=1), rf_predict_optimized.argmax(axis=1)))
    print('\n')

    print("Optimized Model Classification Report")
    print(classification_report(ytest, rf_predict_optimized))
    print('\n')

    print("Optimized Model All Cross Validation Scores")
    print(rfc_cv_score_optimized)
    print('\n')

    print("Optimized Model Mean Cross Validation Score")
    print(rfc_cv_score_optimized.mean())


################################################################################################################

    # # setting up de X test comments to test the algorithm
    # df_test = df_test.sample(n=25000, random_state=33)
    # Xtest = word2Vec(df_test)

    #evaluates the random forest model by saving its predicitons to a .csv file
    # randomForestEvaluator(rf_model, df_train, Xtest)

    return 0


#cleaning the available text to get clean data
def cleanText(text):

    #initialize stopwords
    stopw = stopwords.words('english')

    # remove 'not' from the stopwords while this can negate an insult
    # decided to add 'you' while in toxic conversations you is used to enhance te meaning
    #TODO: enhance comments
    stopw.remove('not')
    stopw.remove('you')
    stopw.remove('your')
    stopw.remove('you\'re')
    stopw.remove('are')

    #make the whole text lowercase so we don't make differences between capitalization
    text = text.lower()

    #subbing words to match cleaner words.
    text = re.sub("\'s", " ", text)
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'n't ", " not ", text, flags=re.IGNORECASE)
    text = re.sub(" n't ", " not ", text, flags=re.IGNORECASE)
    text = re.sub("I'm", "I am", text)
    text = re.sub("shouldn\'t", " should not ", text, flags=re.IGNORECASE)
    text = re.sub("were\'nt", " were not ", text, flags=re.IGNORECASE)
    text = re.sub("can't", " can not ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("\'ll", " will ", text)

    #remove all the stopwords and remove all non letters
    words = word_tokenize(text)
    tokens = [word for word in words if word not in stopw]

    #remove all non letters in the dataset
    tokens = [word for word in tokens if re.match(r'[^\W\d]*$', word)]

    #remove all URLs in the dataset
    # TODO: write a regex for this

    text = ' '.join(tokens)

    #remove punctuation
    text = ''.join([word for word in text if word not in punctuation])

    #dealing with empty data line
    if type(text) != str or text == '':
        return ''


    cleaned_text = text
    return cleaned_text


#making a predictor for the test data
def randomForestEvaluator(randomForest, dataset, Xtest):

    #predict first 200
    predictions = randomForest.predict(Xtest[:200])

    ID = dataset["id"].values
    Classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df_predictions = pd.DataFrame(data=predictions + Classes)
    df_predictions["id"] = ID
    df_predictions = df_predictions[["ID"] + Classes]

    df_predictions.to_csv('Data/predictions.csv')
    return 0


def averageVecValue(comment, model, vectorSize, vocab):
    Vector = np.zeros(vectorSize)

    for word in comment:
        if word in vocab:
            Vector += np.array(model.wv.get_vector(word))

    Vector_value = np.divide(Vector, vectorSize)

    return Vector_value.tolist()


# Vectorizes the data so that i can be used for the random forrest classifier
def word2Vec(dataSet):
    dataSet['comment_text_tokenized'] = dataSet['comment_text'].apply(word_tokenize)
    tokens = dataSet['comment_text_tokenized']

    vectorSize = 300
    word2vec = Word2Vec(tokens, min_count=2, size=vectorSize)
    vocab = word2vec.wv.vocab

    vectorizedData = []
    for index, row in dataSet.iterrows():
        vectorizedData.append(averageVecValue(row['comment_text'], word2vec, vectorSize, vocab))

    vectorizedData = tuple(vectorizedData)

    return vectorizedData


if __name__ == "__main__":
        main()
