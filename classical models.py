#!/usr/bin/env python

'''Reads data into training and test split, performs grid search to find best parameters for each classifier, trains classifier, and prints evaluation report in a review topic classification task.'''

import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-clf", "--classifier", default='nb', type=str,
                        help="Select which classifier is used")
    parser.add_argument("-tp", "--tune_parameters", action="store_true",
                        help="Tune model parameters and report optimal values")
    args = parser.parse_args()
    return args


def read_file(file_path):
    '''read corpus into python and split feats and labels into seperate lists'''
    documents = []
    labels = []
    with open(file_path, encoding='utf-8') as in_file:
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
        rf = {
            'cls__n_estimators': [300],
            'cls__max_depth': [3,5,7],
            'cls__min_samples_leaf':[1,2,3],
            'cls__min_samples_split':[1,3,5]
        }, # Optimal parameter values: {'cls__max_depth': 7, 'cls__min_samples_leaf': 1, 'cls__min_samples_split': 5, 'cls__n_estimators': 300}
        dt = {
            'cls__max_depth': [3, 5, 7, 9],
            'cls__min_samples_leaf': [5, 10, 20, 50, 100],
            'cls__criterion': ["gini", "entropy"]
        }, # Optimal parameter values: {'cls__criterion': 'gini', 'cls__max_depth': 9, 'cls__min_samples_leaf': 10}
        knn = {
            'cls__n_neighbors': [4, 8, 16, 24, 32, 64],
            'cls__weights': ["uniform", "distance"]
        }, # Optimal parameter values: {'cls__n_neighbors': 32, 'cls__weights': 'distance'}
        svc = {
            'cls__C': [.1, 1, 10],
        }, # Optimal parameter values: {'cls__C': .1}
        nb = {
            'cls__alpha': [.00001, .0001, .001, .1, 1, 10, 100,1000],
        } # Optimal parameter values: {'cls__alpha': .1}
    )[model_name]

def tune_parameters(classifier, param_grid, X, Y):
    '''Runs gridsearch for given classifier'''
    print("Performing grid search...")

    grid_search = GridSearchCV(classifier, param_grid, n_jobs=-1)
    grid_search.fit(X, Y)

    print(grid_search.cv_results_)
    print("\nOptimal parameter values:")
    print(grid_search.best_params_)

if __name__ == "__main__":
    args = create_arg_parser()

    # split train data into feats and labels
    X_train, Y_train = read_file("train.tsv")
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=(1,4), min_df=4, max_df=0.70)
    else:
        print("YES!")
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    model = {
        'nb': MultinomialNB(alpha = 1000),
        'svc': LinearSVC(C=.1),
        'knn': KNeighborsClassifier(n_neighbors = 32, weights = "distance"),
        'dt': DecisionTreeClassifier(criterion = 'gini', max_depth = 9, min_samples_leaf = 10),
        'rf': RandomForestClassifier(max_depth = 7, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 300),
        'vote': VotingClassifier(estimators=[('nb', MultinomialNB(alpha = .1)), ('svc', LinearSVC(C=.1))], voting='hard')
    }[args.classifier]

    # defines pipeline including the selected vectorizer and classifier
    classifier = Pipeline([('vec', vec), ('cls', model)])

    if args.tune_parameters:
        tune_parameters(classifier, param_grid(args.classifier), X_train, Y_train)
    else:
        scores = cross_val_score(classifier, X_train, Y_train, cv=9, n_jobs=-1)
        print(f"Average accuracy: {scores.mean()}")
        print(f"Standard deviation: {scores.std()}")
'''
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file, args.sentiment)

        # fits classifier on train data
        classifier.fit(X_train, Y_train)

        # predicts class labels for each feat in X_test
        Y_pred = classifier.predict(X_test)

        # calculates accuracy of predicted labels compared to true labels, and prints classification report
        acc = accuracy_score(Y_test, Y_pred)
        print(f"Final accuracy: {acc}")
        print(classification_report(Y_test, Y_pred))
'''
