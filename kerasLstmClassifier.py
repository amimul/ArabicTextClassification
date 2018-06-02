from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.activations import sigmoid
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint, CSVLogger

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, 64, input_length=reviews.getMaxSenLen(), mask_zero=True))
model.add(GRU(100, return_sequences=True))
model.add(GRU(50, return_sequences=False))
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='./modelChk.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit(xTrain, yTrain, epochs=40, validation_data=(xTest, yTest), batch_size=128, callbacks=[checkpointer, CSVLogger('./LSTMTrain.log')])

model.save('LSTMClassifier.h5')
print(model.evaluate(xTrain, yTrain, batch_size=100))
print(model.evaluate(xTest, yTest, batch_size=100))
