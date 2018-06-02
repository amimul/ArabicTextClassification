import numpy as np
from DataReader import ReviewsReader

np.random.seed(1)

reviews = ReviewsReader()

print('reading embedding')

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

with open('embeddingWords.txt', 'w', encoding='utf8') as f:
    for word in embeddings_index:
        f.write(word + '\n')

print('Found %s word vectors in embedding.' % len(embeddings_index))

word_index = reviews.getTokenizerWordIndex()
with open('trainVocab.txt', 'w', encoding='utf8') as f:
    for word, i in word_index.items():
        f.write(word + '\n')