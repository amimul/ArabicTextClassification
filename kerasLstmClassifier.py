import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras.activations import softmax
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, CSVLogger

import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=False, balanced=False)

yTrain = keras.utils.to_categorical(yTrain - 1)
yTest = keras.utils.to_categorical(yTest - 1)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, 64, input_length=reviews.getMaxSenLen(), mask_zero=False))
model.add(Dropout(0.2))
model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling1D())
model.add(Dense(20, activation='elu'))
model.add(Dense(5, activation=softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='./modelChk.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit(xTrain, yTrain, epochs=40, validation_data=(xTest, yTest), batch_size=128, callbacks=[checkpointer, CSVLogger('./CNNDropoutTrain.log')])

model.save('LSTMClassifier.h5')
print(model.evaluate(xTrain, yTrain, batch_size=32))
print(model.evaluate(xTest, yTest, batch_size=32))
