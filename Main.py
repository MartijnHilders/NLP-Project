import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import nltk
import string
import Data
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from string import punctuation
from gensim.models import Word2Vec



def main():

    # importing the training data using pandas.
    df_train = pd.read_csv('Data/train.csv')

    # save the original text for convenience
    # df_train['comment_text'].to_csv('Data/OriginalText.csv')

    # save the cleaned text for convenience
    # df_train['comment_text'].apply(cleanText).to_csv('Data/cleanedText.csv')

    df_train['comment_text'] = df_train['comment_text'].apply(cleanText)
    print(df_train)



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

    return cleaned_text

def trainingAlgorithm():


    #TODO: get the cleaned training data through the algorithm  trainign it with the accompanied labels


    return 0


if __name__ == "__main__":
        main()