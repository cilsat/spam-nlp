#!/usr/bin/python

import os, sys, string
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

"""
k-fold cross validation
receives pandas dataframe of complete dataset and partitions it into k training and k testing dataframes
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

def train_test(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # unpack arguments and make train/test data/label dicts/lists
    train, test, features, classifier = args

    # create tf idf spare matrix from training data
    if features == 'tfidf':
        fe = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        trainfe = fe.fit_transform(train['data'])

    # train multinomial nb classifier on training data
    if classifier == 'mnb':
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(trainfe, train['labels'])
    elif classifier == 'svm':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(alpha=1e-3, random_state=42).fit(trainfe, train['labels'])
        #clf = SVC().fit(trainfe, train['labels'])

    # extract features from test data
    feats = fe.transform(test['data'])
    # use trained classifier to generate class predictions from test features
    hyp = clf.predict(feats)

    # compare predictions with test labels
    score = np.mean(hyp == test['labels'])
    print(score)

    return score

def train_test_parallel(train_test_sets, features='tfidf', classifier='mnb'):
    from multiprocessing import Pool

    p = Pool()
    scores = p.map(train_test, [(train, test, features, classifier) for train, test in train_test_sets])

    return scores

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        classifier = args[1]
        n_k = int(args[2])
    else:
        classifier = 'mnb'
        n_k = 8

    print(classifier, n_k)

    # load the enron dataframe
    df = load_enron('df-enron.pickle')
    # partition into k sets for cross validation
    ksets = kcv(df, n_k)
    # train and test the k sets in parallel
    scores = train_test_parallel(ksets, features='tfidf', classifier=classifier)

