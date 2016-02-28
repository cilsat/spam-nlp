#!/usr/bin/python

import os, sys, string
from itertools import chain
import cPickle as pickle
import multiprocessing as mp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

import nltk
#from snowballstemmer import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from functools32 import lru_cache

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))

max_terms = 11500

#stemmer = EnglishStemmer()
lmtzr = WordNetLemmatizer()
#stemWords = lru_cache(maxsize=max_terms)(stemmer.stemWords)
lemmatize = lru_cache(maxsize=max_terms)(lmtzr.lemmatize)

# ========================================================================= #
"""
read all text documents in enron and save as pandas table.
columns of table are [file], [text], and [label].
"""
def save_enron(path='/home/cilsat/dev/nlp', form='dataframe'):
    enron_data = {}
    enron_label = {}

    e_dirs = [e for e in os.listdir(path) if e.startswith('enron')]
    for dirs in e_dirs:
        data, labels = build_token_dict(os.path.join(path, dirs), getlabels=True)
        enron_data.update(data)
        enron_label.update(labels)

    if form == 'dataframe':
        import pandas as pd
        df = pd.DataFrame([enron_data, enron_label]).T
        df.columns = ['data', 'labels']

        with open('df-enron.pickle', 'wb') as f:
            pickle.dump(df, f, protocol=2)

def load_enron(path='/home/cilsat/dev/nlp/df-enron.pickle'):
    with open(path, 'rb') as f:
        df = pickle.load(f)
        return df.iloc[:-1]

def build_token_dict(dirpath, getlabels=False):
    print(dirpath)
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
    tokens = text.split()
    #tokens = stemmer.stemWords(tokens)
    tokens = [lemmatize(w) for w in tokens]
    return tokens

# ========================================================================= #
"""
k-fold cross validation
receives pandas dataframe of complete dataset and partitions it into k training and k testing dataframes
"""
def kcv(dataframe, k):
    import pandas as pd
    from sklearn.cross_validation import KFold

    dataframe['labels'] = dataframe['labels'].map({'ham':0, 'spam':1}).astype(bool)

    train_test_sets = []
    
    kf = KFold(len(dataframe), n_folds=k)
    for train, test in kf:
        df_train = dataframe.iloc[train]
        df_test = dataframe.iloc[test]
        train_test_sets.append([df_train, df_test])

    return train_test_sets

def train_test(args):
    
    # unpack arguments and make train/test data/label dicts/lists
    train, test, features, classifier = args

    # create tf idf spare matrix from training data
    if features == 'tfidf':
        fe = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=1290)
        trainfe = fe.fit_transform(train['data'])
    elif features == 'dict':
        fe = CountVectorizer(tokenizer=tokenize, stop_words='english', binary=True)
        trainfe = fe.fit_transform(train['data'])
    elif features == 'lsa':
        svd = TruncatedSVD(n_components=100, random_state=42)
        fe = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.115, max_features=11500)
        trainfe = svd.fit_transform(fe.fit_transform(train['data']))
    elif features == 'rule':
        hamfe = CountVectorizer(tokenizer=tokenize, stop_words='english', max_features=1150)
        spamfe = CountVectorizer(tokenizer=tokenize, stop_words='english', max_features=1150)
        hamfit = hamfe.fit_transform(train['data'].loc[train['labels'] == 0])
        spamfit = spamfe.fit_transform(train['data'].loc[train['labels'] == 1])

    # train multinomial nb classifier on training data
    if classifier == 'mnb':
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(trainfe, train['labels'])
    elif classifier == 'gnb':
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB().fit(trainfe.toarray(), train['labels'])
    elif classifier == 'svm':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='squared_hinge', penalty='l2').fit(trainfe, train['labels'])
    elif classifier == 'log':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log', penalty='l2').fit(trainfe, train['labels'])
    elif classifier == 'rule':
        hamfeats = hamfe.transform(test['data'])
        spamfeats = spamfe.transform(test['data'])
        hyp = np.array(hamfeats.sum(axis=1) < spamfeats.sum(axis=1)).reshape(-1).T
        
    # extract features from test data
    if features == 'lsa':
        feats = svd.transform(fe.transform(test['data']))
    else:
        feats = fe.transform(test['data'])
    # use trained classifier to generate class predictions from test features
    if classifier == 'gnb':
        hyp = clf.predict(feats.toarray())
    elif classifier == 'rule':
        pass
    else:
        hyp = clf.predict(feats)

    # compare predictions with test labels
    score = np.mean(hyp == test['labels'])

    return score

def train_test_parallel(train_test_sets, features='tfidf', classifier='mnb'):
    pool = mp.Pool(mp.cpu_count())
    args = []
    for train, test in train_test_sets:
        args.append([train, test, features, classifier])
    scores = pool.map(train_test, args)

    print(scores)
    print('mean: ' + str(np.mean(scores)))
    print('var: ' + str(np.var(scores)) + '\n')

    return scores

# ========================================================================= #
def save_features(args):
    train, test, n, features = args
    
    if features == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        fe = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_df=0.115, max_features=max_terms)
        train_feats = fe.fit_transform(train['data'])
        test_feats = fe.transform(test['data'])
    
    with open(features + '/k' + str(n) + '.dev', 'wb') as f:
        pickle.dump([train_feats, test_feats, train['labels'], test['labels']], f)

def save_features_parallel(ksets, features='tfidf'):
    pool = mp.Pool(mp.cpu_count())
    pool.map(save_features, [[ksets[n][0], ksets[n][1], n, features] for n in range(len(ksets))])

def load_features(args):
    path, n = args
    train_feats, test_feats, train_labels, test_labels = pickle.load(open(os.path.join(path, 'k' + str(n) + '.dev'), 'rb'))
    return [train_feats, test_feats, train_labels, test_labels]

def fast_train_test(args):
    train_feats, test_feats, train_labels, test_labels, classifier, features = args

    if classifier == 'mnb':
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(train_feats, train_labels)
    elif classifier == 'svm':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier().fit(train_feats, train_labels)

    hyp = clf.predict(test_feats)
    score = hyp == test_labels
    avg = np.mean(score)
    var = np.var(score)
    
    return avg

"""
train and test from saved tf-idf sparse matrices. path refers to location of .tfidf files
"""
def fast_train_test_parallel(path='.', classifier='mnb', features='tfidf'):
    pool = mp.Pool(mp.cpu_count())
    files = os.listdir(os.path.join(os.path.abspath(path), features))
    kset = pool.map(load_features, [[os.path.join(path, features), n] for n in range(len(files))])
    scores = pool.map(fast_train_test, [[train_feats, test_feats, train_labels, test_labels, classifier, features] for train_feats, test_feats, train_labels, test_labels in kset])
    pool.close()
    pool.terminate()
    pool.join()

    return scores

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        n_k = int(args[1])
        feats = args[2]
        classifier = args[3]
    else:
        classifier = 'mnb'
        n_k = 12
        feats = 'tfidf'

    print(feats + ' + ' + classifier)

    # load the enron dataframe
    df = load_enron('df-enron.pickle')
    # partition into k sets for cross validation
    ksets = kcv(df, n_k)
    # train and test the k sets in parallel
    scores = train_test_parallel(ksets, features=feats, classifier=classifier)
