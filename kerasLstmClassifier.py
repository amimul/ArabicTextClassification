from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.activations import sigmoid
from keras.layers.recurrent import GRU

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)


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


model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=reviews.getMaxSenLen(), mask_zero=True, trainable=False))
model.add(GRU(10, return_sequences=False))
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(xTrain, yTrain, epochs=2)

model.save('LSTMClassifier.h5')
print(model.evaluate(xTrain, yTrain))
print(model.evaluate(xTest, yTest))
