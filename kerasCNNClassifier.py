from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, Dropout
from keras.activations import sigmoid
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint, CSVLogger

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

EMBEDDING_DIM = 64

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, EMBEDDING_DIM, input_length=reviews.getMaxSenLen(), mask_zero=False))
model.add(Dropout(0.4))
model.add(Conv1D(50, 5, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='./modelChk.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit(xTrain, yTrain, epochs=40, validation_data=(xTest, yTest), batch_size=128, callbacks=[checkpointer, CSVLogger('./CNNDropoutTrain.log')])

print('train loss and Acc', model.evaluate(xTrain, yTrain, batch_size=128))
print('test loss and Acc', model.evaluate(xTest, yTest, batch_size=128))
