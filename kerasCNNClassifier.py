from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D
from keras.activations import sigmoid
from keras.layers.recurrent import GRU

from keras.callbacks import ModelCheckpoint, CSVLogger

import numpy as np
from DataReader import ReviewsReader
import gensim

np.random.seed(1)

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

## read and prepare embedding
embeddingModel = gensim.models.Word2Vec.load('./www_cbow_100/www_cbow_100')

word_index = reviews.getTokenizerWordIndex()
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
c = [0, 0]
for word, i in word_index.items():
    word = ReviewsReader.cleanStr(word)

    if word in embeddingModel.wv.vocab:
        embedding_vector = embeddingModel.wv[word]
        embedding_matrix[i] = embedding_vector
        c[0] += 1
    else:
        c[1] += 1
print('found', c)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(reviews.getVocabSize() + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=reviews.getMaxSenLen(), mask_zero=False, trainable=False))
model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(MaxPool1D(5))
model.add(Conv1D(50, 5, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='./modelChk.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit(xTrain, yTrain, epochs=40, validation_data=(xTest, yTest), batch_size=128, callbacks=[checkpointer, CSVLogger('./CNNDropoutTrain.log')])

model.save('CNNClassifier.h5')
print(model.evaluate(xTrain, yTrain, batch_size=128))
print(model.evaluate(xTest, yTest, batch_size=128))
