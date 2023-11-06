#!/usr/bin/env python

import random as python_random
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.utils import to_categorical

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.tsv', type=str,
                        help="Input file to learn from (default train.tsv)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.tsv',
                        help="Separate dev set to read in (default dev.tsv)")
    parser.add_argument("-t", "--test_file", type=str, default='test.tsv',
                        help="If added, use trained model to predict on test set (default test.tsv)")
    parser.add_argument("-l", "--language", type=str, default='en',
                        help="Choose which language model to use. Options: en, multi, nl, zh (chinese), ja, ko, ge, it, es (spanish), sv (swedish)")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])
    return documents, labels


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here'''
    # Transform set labels
    Y_train = to_categorical(Y_train)
    Y_dev = to_categorical(Y_dev)

    # You can fine-tune the batch size and epochs here
    loss_function = BinaryCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=5e-5)
    verbose = 1
    batch_size = 64
    epochs = 3
    # Early stopping: stop training if there are three consecutive epochs without improvement
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Compile and train the model on our data
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size,
              validation_data=(X_dev, Y_dev))

    # Print the final accuracy of the model for a clearer summary
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, identifier):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Use the trained model for prediction
    Y_pred = model.predict(X_test).logits
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    # Map class labels to class names
    label_to_name = {0: 'NOT', 1: 'OFF'}
    Y_pred = [label_to_name[label] for label in Y_pred]
    Y_test = [label_to_name[label] for label in Y_test]

    class_report = classification_report(Y_test, Y_pred)

    # Print the classification report and accuracy
    print(class_report)
    print('Accuracy on our own {} set: {}'.format(identifier, round(accuracy_score(Y_test, Y_pred), 3)))


def main():
    '''Main function for training and testing a neural network based on command-line arguments'''
    args = create_arg_parser()

    # Read data
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)


    # choose model
    model_dict = {"en": "microsoft/deberta-v3-base",
                  "multi": "bert-base-multilingual-uncased",
                  "nl": "bert-base-dutch-cased",
                  "zh": "bert-base-chinese",
                  "ja": "cl-tohoku/bert-base-japanese-v3",
                  "ko": "kykim/bert-kor-base",
                  "ge": "bert-base-german-cased",
                  "it": "dbmdz/bert-base-italian-cased",
                  "es": "dccuchile/bert-base-spanish-wwm-uncased",
                  "sv": "KB/bert-base-swedish-cased"}

    # Tokenization and data preparation code blocks here
    # Tokenize and prepare input data
    lm = model_dict[args.language]
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # Tokenize the training data
    tokens_train = tokenizer(X_train, padding=True, max_length=100, truncation=True, return_tensors="np").data

    # Tokenize the development data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=100, truncation=True, return_tensors="np").data

    # Create the model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    # Perform one-hot encoding of labels
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)

    # Train the model
    model = train_model(model, tokens_train, Y_train_bin, tokens_dev, Y_dev_bin)

    # Make predictions on the specified test set
    if args.test_file:
        # Read the test set and tokenize and prepare it
        X_test, Y_test = read_corpus(args.test_file)
        tokens_test = tokenizer(X_test, padding=True, max_length=100, truncation=True, return_tensors="np").data
        attention_mask_test = tokens_test['attention_mask']
        Y_test_bin = encoder.transform(Y_test)
        Y_test_bin = to_categorical(Y_test_bin)
        # Make predictions
        test_set_predict(model, tokens_test, Y_test_bin, "test")

if __name__ == '__main__':
    main()
