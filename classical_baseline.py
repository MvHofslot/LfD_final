#!/usr/bin/env python

import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
import spacy

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.tsv', type=str,
                        help="Train file to learn from (default ../data/train.tsv)")
    parser.add_argument("-df", "--test_file", default='dev.tsv', type=str,
                        help="Test file to evaluate on")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-u", "--union", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-clf", "--classifier", default='nb', type=str,
                        help="Select which classifier is used")
    parser.add_argument("-tp", "--tune_parameters", action="store_true",
                        help="Tune model parameters and report optimal values")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''read corpus into python and split feats and labels into seperate lists'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


def param_grid(model_name):
    return dict(
        random_forest = {
            'n_estimators': [50,100,150,200],
            'max_features': [0.1, 0.25, 0.5, 1],
            'max_depth': [3, 5, 7, 9],
            'max_samples': [0.3, 0.5, 0.8]
        },
        decision_tree = {
            'max_depth': [3, 5, 7, 9],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'criterion': ["gini", "entropy"]
        },
        knn = {
            'cls__n_neighbors': [4, 8, 12, 16, 20, 24, 32, 64],
            'cls__weights': ["uniform", "distance"]
        },
        svc = {
            'cls__C': [0.1, 1, 10], 
            'cls__gamma': [1, 0.1, 0.01, 0.001],
            'cls__kernel': ['rbf', 'linear']
        }
    )[model_name]


def tune_parameters(classifier, param_grid, X, Y):
    print("Performing grid search...")

    grid_search = GridSearchCV(classifier, param_grid, n_jobs=-1)
    grid_search.fit(X, Y)

    print(grid_search.cv_results_)
    print("\nOptimal parameter values:")
    for param, value in grid_search.best_params_:
        print(f"{param}: {value}")


def spacy_pos(txt):
    return [token.pos_ for token in nlp(txt)]


if __name__ == "__main__":
    args = create_arg_parser()

    X_train, Y_train = read_corpus(args.train_file)
    print("OFF: ", Y_train.count("OFF"), "NOT: ", Y_train.count("NOT"))

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(1,5), max_df=0.9)
    elif args.union:
        tfidf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(1,5), max_df=0.9)
        char = CountVectorizer(analyzer='char_wb', ngram_range=(3,6), min_df=0.25)
        
        #nlp = spacy.load("en_core_web_sm")
        #pos = CountVectorizer(tokenizer=spacy_pos)

        vec = FeatureUnion([("tfidf", tfidf), ("chars", char)]) #("pos", pos)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
    

    # Combine the vectorizer with a Naive Bayes classifier
    # Of course you have to experiment with different classifiers
    # You can all find them through the sklearn library
    model = {
        'nb': MultinomialNB(),
        'svc': SVC(kernel = 'linear'),
        'knn': KNeighborsClassifier(),
        'dt': DecisionTreeClassifier(),
        'rf': RandomForestClassifier()
    }[args.classifier]
    classifier = Pipeline([('vec', vec), ('cls', model)])
    print(vec,model)

    if args.tune_parameters:
        tune_parameters(classifier, param_grid(args.classifier), X_train, Y_train)
    classifier.fit(X_train, Y_train)
    X_test, Y_test = read_corpus(args.test_file)

    Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print(f"Final accuracy: {acc}")
    print(classification_report(Y_test, Y_pred))