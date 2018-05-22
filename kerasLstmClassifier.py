from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.activations import sigmoid
from keras.optimizers import rmsprop
from keras.losses import mse
from keras.layers.recurrent import LSTM

import numpy as np

np.random.seed(1)

VOCAB_SIZE = 5
MAX_SEN_LEN = 4
N_CLASSES = 2
N_SEN = 2
x = np.random.randint(VOCAB_SIZE + 1, size=(N_SEN, MAX_SEN_LEN))
y = np.random.randint(N_CLASSES, size=(N_SEN,))
print(x)
print(y)

model = Sequential()
## we use mask zero as we deal with different len sentences so we pad with zeros
model.add(Embedding(VOCAB_SIZE, 10, input_length=MAX_SEN_LEN, mask_zero=True))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1, activation=sigmoid))
model.compile(rmsprop(), mse)
model.summary()
model.fit(x, y)

model.save('LSTMClassifier.h5')
print(model.evaluate(x, y))
