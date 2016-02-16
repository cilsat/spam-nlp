#!/usr/bin/python

import os
import string
from itertools import chain
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
stemmer = PorterStemmer()

"""
read all text documents in enron and save as pandas table.
columns of table are [file], [text], and [label].
"""
def save_enron(path='/home/cilsat/dev/nlp', form='dataframe'):
    enron_data = {}
    enron_label = {}

    e_dirs = [e for e in os.listdir(path) if e.startswith('enron')]
    for dirs in e_dirs:
        data, labels = build_token_dict(dirs, getlabels=True)
        enron_data.update(data)
        enron_label.update(labels)

    if form == 'dataframe':
        import pandas as pd
        df = pd.DataFrame([enron_data, enron_label]).T
        df.columns = ['data', 'labels']

        import pickle
        with open('df-enron.pickle', 'wb') as f:
            pickle.dump(df, f, protocol=2)

def load_enron(path='/home/cilsat/dev/nlp/df-enron.pickle'):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_token_dict(dirpath, getlabels=False):
    token_dict = {}
    labels = {}
    for subdir, dirs, files in os.walk(dirpath):
        subname = subdir.split('/')[-1]
        files.sort()
        for file in files:
            file_path = subdir + os.path.sep + file
            file_text = open(file_path, 'r').read()
            token_dict[file] = prep_text(file_text)
            labels[file] = subname

    if getlabels:
        return token_dict, labels
    else:
        return token_dict

def prep_text(text):
    endl = text.replace('\r','').replace('\n',' ').lower()
    alph = ''.join([i for i in endl if i in string.printable[10:36]+' '])
    #tokens = nltk.word_tokenize(text)
    return alph

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = [stemmer.stem(w) for w in tokens]
    return stems

def count_tokens(tokens):
    from collections import Counter
    return Count(tokens)

def tfidf(data_dict):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(data_dict.values())
    return tfidf

def eval(text, tfidf):
    response = tfidf.transform([text])
    #feats = tfidf.get_feature_names()
    #for i in response.nonzero()[1]:
    #    print(feats[i], '\t', response[0,i])
    return response

def mnb(data_dict, label_dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    fe = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    trainfe = fe.fit_transform(data_dict.values())
    clf = MultinomialNB().fit(trainfe, label_dict.values())
    return clf, fe

"""
receives pandas dataframe with 3 columns and outputs k training and k testing dataframes
"""
def kcv(dataframe, k):
    import pandas as pd
    from sklearn.cross_validation import KFold

    dataframe['labels'] = dataframe['labels'].map({'ham':0, 'spam':1}).astype(bool)
    #n_spam = len(dataframe.loc[dataframe['labels'] == 1)])
    #n_ham = len(dataframe.loc[dataframe['labels'] == 0)])

    train_test_sets = []
    
    kf = KFold(len(dataframe), n_folds=k)
    for train, test in kf:
        df_train = dataframe.iloc[train]
        df_test = dataframe.iloc[test]
        train_test_sets.append([df_train, df_test])

    return train_test_sets

def train_test(train_set, test_set, scores, method='mnb'):
    train_data_dict = dict(train_set['data'])
    train_label_dict = dict(train_set['labels'])
    test_data = test_set['data'].tolist()
    test_label = np.array(test_set['labels'])

    # train multinomial nb classifier on training data
    clf, fe = mnb(train_data_dict, train_label_dict)
    # extract features from test data
    feats = fe.transform(test_data)
    # use trained classifier to generate class predictions from test features
    hyp = clf.predict(feats)
    # compare predictions with test labels
    stack = np.vstack((hyp, test_label)).T
    diffs = np.diff(stack, axis=-1)
    score = 100 - np.sum(diffs)*100./diffs.size
    scores.append(score)

def train_test_parallel(train_test_sets, method='mnb'):
    from multiprocessing import Process

    processes = []
    scores = []
    for train_set, test_set in train_test_sets:
        p = Process(target=train_test, args=(train_set, test_set, scores, method))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return np.mean(scores)

if __name__ == "__main__":
    spam = build_token_dict(os.path.join(path, 'enron1/spam'))
    ham = buil_token_dict(os.path.join(path, 'enron1,ham'))
    sham = dict(chain(ham.iteritems(), spam.iteritems())) 

    spamtf = tfidf(spam)
    hamtf = tfidf(ham)
