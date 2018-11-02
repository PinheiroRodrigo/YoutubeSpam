import csv
import re
import numpy as np
from pandas import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split


def main():
    matrix = matrix_from_file("Eminem.csv")
    matrix = clean_and_put_length(matrix)
    dictionary = create_dictionary(matrix)
    print(dictionary)
    # Separe between train and test
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
def create_dictionary(matrix):
    comment_list = matrix[:, 1]
    output = np.concatenate(comment_list).ravel().tolist()[0]
    # Tokenize
    cv = CountVectorizer()
    ex = cv.fit_transform(output)
    return cv.get_feature_names()


main()
#print(avg_len)
#print(pandas.DataFrame(matrix))
