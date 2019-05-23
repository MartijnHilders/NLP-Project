import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import nltk
import string
import Data
import numpy
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from gensim.models import Word2Vec


def main():
    cleanAndTokenize()
    return 0





def cleanAndTokenize():
    text = 1
    print("hello")
    cleaned_text = text

    return cleaned_text

if __name__ == "__main__":
        main()