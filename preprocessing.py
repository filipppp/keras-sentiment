import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import hickle

TAG_REG = re.compile(r'<[^>]+>')


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def remove_tags(text):
    return TAG_REG.sub('', text)


if __name__ == "__main__":
    mov_reviews = pd.read_csv("imdb.csv")

    # are any null values?
    print(mov_reviews.isnull().values.any())

    # plot occurrences of each sentiment
    print(mov_reviews.head())
    sns.countplot(x="sentiment", data=mov_reviews)
    plt.show()

    # create tarining data

    X = []
    sentences = list(mov_reviews["review"])
    for sen in sentences:
        X.append(preprocess_text(sen))

    # print preprocessed review
    print(X[2])

    # create targets
    y = mov_reviews["sentiment"]
    y = [1 if sentiment == "positive" else 0 for sentiment in y]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    print(X_train[0])

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # set max input length
    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # create word to vector dictionary from glove
    embeddings_dictionary = dict()
    glove_file = open("glove.6B.100d.txt", encoding="utf8")
    for line in glove_file:
        line = line.split(" ")

        word = line[0]
        vector = line[1:]

        embeddings_dictionary[word] = np.asarray(vector, dtype="float32")
    glove_file.close()

    # create embedding matrix
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    hickle.dump((X_train, X_test, y_train, y_test, vocab_size, embedding_matrix, maxlen, tokenizer), "preprocessed.hickle")
