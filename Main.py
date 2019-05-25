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

    # test on smallportion
    df_train = df_train.sample(n=10000, random_state=33)

    # setting up the X training comments (vectorize them to be able to be used as input for model) and Y training labels
    print("setting up training data ")
    Xtrain = word2Vec(df_train)
    Ytrain = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]


    # importing the test data set to test the algorithm
    df_test = pd.read_csv('Data/test.csv')

    # setting up de X test comments to test the algorithm
    df_test = df_test.sample(n=10000, random_state=33)
    Xtest = word2Vec(df_test)

    #run the algorithm for the first time and get an idea of the accuracy with the basic parameters.
    print("started training the model")
    rf_model = RandomForestClassifier()
    rf_model.fit(Xtrain, Ytrain)

    # test the accuracy of the model on a split training dataset
    #TODO: this..



    #evaluates the random forest model by saving its predicitons to a .csv file
    randomForestEvaluator(rf_model, df_train, Xtest)

################################################################################################################





    # for faster testing we only take a part of the training set and testing it in a normal randomForest classifier
    # df_train = df_train.sample(n=25000, random_state=33)





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


def trainingAlgorithm(dataset):
    #TODO: get the cleaned training data through the algorithm  trainign it with the accompanied labels
    print('Started training')

    #unigram model
    toxicUnigramCounts = collections.defaultdict(lambda: 0)
    toxicTotal = 0
    neutralUnigramCounts = collections.defaultdict(lambda: 0)
    neutralTotal = 0

    for index, row in dataset.iterrows():
        #print(row['id'])
        for word in word_tokenize(row['comment_text']):
            if row['toxic']==1:
                toxicUnigramCounts[word] = toxicUnigramCounts[word] + 1
                toxicTotal += 1
            if row['toxic']==0:
                neutralUnigramCounts[word] = neutralUnigramCounts[word] + 1
                neutralTotal += 1




    return 0


def unigramtesting(sentence,toxicTotal,neutralTotal,toxicUnigramCounts,neutralUnigramCounts):


    toxic = " False "

    toxicScore = 0.0
    neutralScore = 0.0
    for token in sentence:
        toxicCount = toxicUnigramCounts[token] + 1
        toxicScore += math.log(toxicCount)
        toxicScore -= math.log(toxicTotal)

        neutralCount = neutralUnigramCounts[token] + 1
        toxicScore += math.log(neutralCount)
        toxicScore -= math.log(neutralTotal)


    if toxicScore > neutralScore:
        toxic = " True "
        # return 1
    else:
        toxic = " False "
        # return 0



    print("ToxicScore = " + toxicScore + " Toxic? = " + toxic)


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

    return vectorizedData


if __name__ == "__main__":
        main()
