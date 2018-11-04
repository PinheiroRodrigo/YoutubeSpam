import csv
import re
import numpy as np
from pandas import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
# CLASSIFIERS
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


# TODO: use length of comments in the models

def main(filename):
    matrix = matrix_from_file(filename)
    matrix = clean_and_put_length(matrix)
    ocurrencies, vocabulary, dictionary = bag_of_words(matrix)
    # Separe between train and test (p for params, l for labels)
    (train_p, test_p, train_l, test_l) = (
        train_test_split(ocurrencies, matrix[:, 2].ravel().tolist()[0], test_size=0.3))

    # NAIVE BAYES CLASSIFIER
    nb_score = naive_bayes_classifier(train_p, train_l, test_p, test_l)
    print('NaiveBayes had an accuracy of {:.2%}'.format(nb_score))
    # KNN CLASSIFIER
    knn_2_score = knn_classifier(train_p, train_l, test_p, test_l, 2)
    print('KNN-2 had an accuracy of {:.2%}'.format(knn_2_score))
    knn_3_score = knn_classifier(train_p, train_l, test_p, test_l, 3)
    print('KNN-3 had an accuracy of {:.2%}'.format(knn_3_score))

# Read file and get Matrix
def matrix_from_file(filename):
    reader = csv.reader(open(filename, "r+"), delimiter=",")
    # Transform into matrix
    full_matrix = list(reader)
    # Remove label names
    matrix = np.matrix(full_matrix[1:])
    # Remove unused columns
    return matrix[:, 3:]

# Regex to remove unwanted characters
def clean_and_put_length(matrix):
    regex = re.compile('([^\s\w]|_)+')
    len_column = []
    # Update matrix after regex replace
    for x in matrix[:, 0]:
        x[0, 0] = regex.sub(' ', x[0, 0])
        # add comment length column
        len_column.append(len(x[0, 0]))
    # Transform into numpy array and insert into matrix
    len_column = np.array(len_column)
    matrix = np.insert(matrix, 0, len_column, axis=1)
    return matrix

# Get list of the comments
def bag_of_words(matrix):
    comment_list = matrix[:, 1]
    # Transform into a list with all comments
    corpus = np.concatenate(comment_list).ravel().tolist()[0]
    # Tokenize
    cv = CountVectorizer()
    # Get matrix of ocurrencies
    ocurrencies = cv.fit_transform(corpus).todense()
    vocabulary = cv.vocabulary_
    dictionary = cv.get_feature_names()
    return ocurrencies, vocabulary, dictionary

def naive_bayes_classifier(train_p, train_l, test_p, test_l):
    clf = MultinomialNB()
    clf.fit(train_p, train_l)
    score = clf.score(test_p, test_l)
    return score

def knn_classifier(train_p, train_l, test_p, test_l, neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(train_p, train_l)
    score = knn.score(test_p, test_l)
    return score

# def bag_of_words_example():
#     corpus = ["all my cats", "really good cats"]
#     cv = CountVectorizer()
#     ocurrencies = cv.fit_transform(corpus).todense()
#     vocabulary = cv.vocabulary_
#     print(vocabulary)
#     print(ocurrencies)
#bag_of_words_example()


main("Eminen.csv")
