from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.activations import sigmoid
from keras.layers.recurrent import LSTM

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, 64, input_length=reviews.getMaxSenLen(), mask_zero=True))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(xTrain, yTrain)

model.save('LSTMClassifier.h5')
print(model.evaluate(xTrain, yTrain))
print(model.evaluate(xTest, yTest))
