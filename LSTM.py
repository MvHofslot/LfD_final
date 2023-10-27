#!/usr/bin/env python

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import TextVectorization, Bidirectional
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

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
    parser.add_argument("-e", "--embeddings", default='glove.twitter.27B.200d.txt', type=str,
                        help="Embedding file we are using (default glove.twitter.27B.200d.txt)")
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

def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = {}
    with open(embeddings_file, 'r', encoding='UTF-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def get_emb_matrix(vocab, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    # Get an embedding matrix based on vocabulary and embeddings
    num_tokens = len(vocab) + 2
    word_index = dict(zip(vocab, range(len(vocab))))
    embedding_dim = len(emb["the"])
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_model(Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define model hyperparameters, you can fine-tune them here
    learning_rate = 5e-4
    loss_function = 'binary_crossentropy'
    optim = Adam(learning_rate=learning_rate)
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    units = 128
    dropout = 0.5
    recurrent_dropout = 0.5

    # Create the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))
    model.add(Bidirectional(LSTM(units=units, activation='relu', dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(units=num_labels, activation="sigmoid"))

    # Compile the model
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here'''
    # Transform set labels
    Y_train = to_categorical(Y_train) 
    Y_dev = to_categorical(Y_dev) 

    # You can fine-tune the batch size and epochs here
    verbose = 1
    batch_size = 64
    epochs = 50
    # Early stopping: stop training when there are two consecutive epochs without improving
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model

def test_set_predict(model, X_test, Y_test, identifier):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Use the trained model for prediction
    Y_pred = model.predict(X_test)
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
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)

    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)

    # Adapt the vectorizer to the data to build a vocabulary
    vectorizer.adapt(text_ds)
    vocab = vectorizer.get_vocabulary()

    # Generate an embedding matrix based on the vocabulary and pre-trained word embeddings
    emb_matrix = get_emb_matrix(vocab, embeddings)

    # Initialize a label binarizer to transform categorical labels to binary format
    encoder = LabelBinarizer()

    # Transform string labels to one-hot encodings
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)

    # Create a model
    model = create_model(Y_train, emb_matrix)

    # Vectorize the text data for training and development sets
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        Y_test_bin = to_categorical(Y_test_bin)
        test_set_predict(model, X_test_vect, Y_test_bin, "test")

if __name__ == '__main__':
    main()
