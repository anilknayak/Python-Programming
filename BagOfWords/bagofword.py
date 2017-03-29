import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import collections

train_base_path = "train"
test_base_path = "test"

def train():
    training_class_path = os.listdir(train_base_path)
    training_details = []
    for class_path in training_class_path:
        training = {}
        txt_file = open(train_base_path + "\\" + class_path,"r")
        txt = txt_file.read()
        example1 = BeautifulSoup(txt,"lxml")
        text  = example1.get_text()
        text = text.lower()
        words = text.split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=5000)
        train_data_features = vectorizer.fit_transform(meaningful_words)
        vocab = vectorizer.get_feature_names()
        dist = np.sum(train_data_features, axis=0)

        histogram = collections.Counter(meaningful_words)

        training['class'] = class_path
        training['histogram'] = histogram
        training_details.append(training)
    return training_details

def test(training):
    testing_class_path = os.listdir(test_base_path)
    document_name = ''
    document_word_count = 0
    for class_path in testing_class_path:
        txt_file = open(train_base_path + "\\" + class_path,"r")
        txt = txt_file.read()
        example1 = BeautifulSoup(txt,"lxml")
        text  = example1.get_text()
        text = text.lower()
        words = text.split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=5000)
        train_data_features = vectorizer.fit_transform(meaningful_words)
        vocabs = vectorizer.get_feature_names()

        for train in training:
            docu_count = 0
            class_dtl = train['class']
            histogram_dtl = train['histogram']
            for vocab in vocabs:
                count = histogram_dtl.get(vocab)
                if count != None:
                    docu_count = docu_count + count

            if docu_count > document_word_count:
                document_word_count = docu_count
                document_name = class_dtl

        print('Predicted Class is ' + document_name)

training = train()
#print(training)
test(training)