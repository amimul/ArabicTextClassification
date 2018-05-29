import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout
from keras.activations import softmax
from keras.layers.recurrent import LSTM

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=False, balanced=False)

yTrain = keras.utils.to_categorical(yTrain - 1)
yTest = keras.utils.to_categorical(yTest - 1)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, 64, input_length=reviews.getMaxSenLen(), mask_zero=True))
model.add(LSTM(30, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(10, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(5, activation=softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(xTrain, yTrain, epochs=3, batch_size=64)

model.save('LSTMClassifier.h5')
print(model.evaluate(xTrain, yTrain, batch_size=32))
print(model.evaluate(xTest, yTest, batch_size=32))
