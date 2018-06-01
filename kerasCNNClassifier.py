from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D
from keras.activations import sigmoid
from keras.layers.recurrent import GRU

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)


try:
    embedding_matrix = np.load('emb.npy')
    print('loaded prev saved embedding matrix')
except:
    ## read and prepare embedding
    embeddings_index = {}
    with open('./wiki.ar.vec', 'r', encoding='utf8') as f:
        n, d = map(int, f.readline().split())
        for line in f:
            line = line.strip()
            values = line.split()
            word = ' '.join(values[:-d])
            coefs = np.asarray(values[-d:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors in embedding.' % len(embeddings_index))

    word_index = reviews.getTokenizerWordIndex()
    EMBEDDING_DIM = d
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    np.save('emb', embedding_matrix)

# model = Sequential()
# ## we use mask zero as we deal with different len sentences so we pad with zeros
# model.add(Embedding(reviews.getVocabSize() + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=reviews.getMaxSenLen(), mask_zero=False, trainable=False))
# model.add(Conv1D(128, 5, padding='same', activation='relu'))
# model.add(MaxPool1D(5))
# model.add(Conv1D(50, 5, padding='same', activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(1, activation=sigmoid))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import keras
for i in range(2, 15):
    print('##################')
    print('################## EPOCH', i)
    print('##################')
    
    model = keras.models.load_model('./CNNClassifier_Epoch24_dropout0_2_oneMoreEpoch.h5')
    model.fit(xTrain, yTrain, epochs=1, batch_size=128)

    model.save('./CNNClassifier_Epoch24_dropout0_2_oneMoreEpoch.h5')
    print(model.evaluate(xTrain, yTrain, batch_size=128))
    print(model.evaluate(xTest, yTest, batch_size=128))
