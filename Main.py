import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import nltk
import string
import Data
import numpy as np
import collections
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
import math
from gensim.models import Word2Vec



def main():

    print(wordpunct_tokenize("Cool. Yes, very cool!"))

    # importing the training data using pandas.
    # nltk.download('stopwords')
    # nltk.download('punkt')
    df_train = pd.read_csv('Data/train.csv')

    # save the original text for convenience
    # df_train['comment_text'].to_csv('Data/OriginalText.csv')

    # save the cleaned text for convenience
    # df_train['comment_text'].apply(cleanText).to_csv('Data/cleanedText.csv')
    trainingAlgorithm(df_train)
    df_train['comment_text'] = df_train['comment_text'].apply(cleanText)
    print(df_train['comment_text'])
    return 0




#cleaning the available text to get clean data

def cleanText(text, stem_words=True):

    stopw = stopwords.words('english')
    stopw.extend(',')

    # remove not not from the stopwords while this can negate an insult
    # decided to add you while in toxic conversations you is used to enhance te meaning ....
    #TODO: change
    stopw.remove('not')
    stopw.remove('you')
    stopw.remove('your')
    stopw.remove('you\'re')

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


    #remove all the stopwords
    stopword = word_tokenize(text)
    tokens = [word for word in stopword if word not in stopw]
    text = ' '.join(tokens)

    #remove punctuation
    text = ''.join([word for word in text if word not in punctuation]).lower()

    #dealing with empty questions
    if type(text) != str or text == '':
        return ''



    cleaned_text = text
    #print(cleaned_text)
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
    print(toxicTotal)


    return 0

def unigramtesting(sentence,toxicTotal,neutralTotal,toxicUnigramCounts,neutralUnigramCounts):


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
        return 1
    else:
        return 0







if __name__ == "__main__":
        main()