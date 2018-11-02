import csv
import re
import numpy as np
from pandas import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Read file
def read_file(filename):
    reader = csv.reader(open(filename, "r+"), delimiter=",")
    # Transform into matrix
    full_matrix = list(reader)
    # Remove label names
    matrix = np.matrix(full_matrix[1:])
    # Remove unused columns
    clear_strings(matrix[:, 3:])


# Regex to remove unwanted characters
def clear_strings(matrix):
    regex = re.compile('([^\s\w]|_)+')
    total_len = 0
    for x in matrix[:, 0]:
        x[0, 0] = regex.sub(' ', x[0, 0])
        # Get avg comment length
        total_len += len(x[0])
    avg_len = total_len/len(matrix)
    tokenizer(matrix)

# Get a list of the comments
def tokenizer(matrix):
    comment_list = matrix[:, 0]
    output = np.concatenate(comment_list).ravel().tolist()[0]
    # Tokenize
    cv = CountVectorizer()
    ex = cv.fit_transform(output)
    print(ex.toarray())


read_file("Eminem.csv")
#print(avg_len)
#print(pandas.DataFrame(matrix))
