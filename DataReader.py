import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

class ReviewsReader:
    '''
        reviews.tsv: a tab separated file containing the "cleaned up" reviews. It contains over 63,000 reviews. The format is:

                rating<TAB>review id<TAB>user id<TAB>book id<TAB>review
        where:

                rating: the user rating on a scale of 1 to 5
                review id: the goodreads.com review id
                user id: the goodreads.com user id
                book id: the goodreads.com book id
                review: the text of the review
    '''

    def __init__(self, dataTsvPath = './data/reviews.tsv'):
        self.reviews = pandas.read_csv(dataTsvPath, sep='\t', header=None, names=['rating', 'review_id', 'user_id', 'book_id', 'review'], index_col=None)

        self.reviews['2classRating'] = self.reviews['rating'].apply(lambda x: 1 if x > 3 else 0)

        ### normalizaion
        self.reviews['review'] = self.reviews['review'].apply(lambda x: ReviewsReader.__normalize(x))

    def __normalize(s):
        x = s
        x = re.sub(r'[^\w\s]', '', x)

        x = x.replace('آ', 'ا')
        x = x.replace('أ', 'ا')
        x = x.replace('ة', 'ه')
        x = x.replace('إ', 'ا')
        x = x.replace('ى', 'ي')
        x = x.replace('ﻵ', 'لا')
        x = x.replace('ـ', '')

        x = x.replace('\u0660', '0')
        x = x.replace('\u0661', '1')
        x = x.replace('\u0662', '2')
        x = x.replace('\u0663', '3')
        x = x.replace('\u0664', '4')
        x = x.replace('\u0665', '5')
        x = x.replace('\u0666', '6')
        x = x.replace('\u0667', '7')
        x = x.replace('\u0668', '8')
        x = x.replace('\u0669', '9')

        x = ' '.join(' '.join(re.findall(r'([^\d]+|[\d]+)', x)).split())

        x = re.sub(r'([\u0600-\u06ff])\1{2,}', r'\1', x)
        return x

    def __readIndexedData(self, indexingFile):
        '''
            returns the data in ready-to-consume format by the training model
        '''

        with open(indexingFile, 'r') as f:
            idxs = f.readlines()
            idxs = map(lambda x: x.strip(), idxs)
            idxs = map(int, idxs)
            idxs = list(idxs)
        
        selectedReviews = self.reviews.loc[idxs]

        return selectedReviews

    def readTrainTest(self, twoClass=True, balanced=False):
        UN = '' if balanced else 'un'
        nClass = '2' if twoClass else '5'

        trainDF = self.__readIndexedData('./data/' + nClass + 'class-' + UN + 'balanced-train.txt')
        testDF = self.__readIndexedData('./data/' + nClass + 'class-' + UN + 'balanced-test.txt')
        
        ### featurization
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(trainDF['review'].tolist())
        trainDF['feat'] = trainDF['review'].apply(lambda x: self.tokenizer.texts_to_sequences([x])[0])
        testDF['feat'] = testDF['review'].apply(lambda x: self.tokenizer.texts_to_sequences([x])[0])

        ### padding
        self.maxSenLen = 100
        trainDF['feat'] = trainDF['feat'].apply(lambda x: pad_sequences([x], maxlen=self.maxSenLen, padding='post', value=0)[0])
        testDF['feat'] = testDF['feat'].apply(lambda x: pad_sequences([x], maxlen=self.maxSenLen, padding='post', value=0)[0])

        ratingCol = '2classRating' if twoClass else 'rating'
        return map(lambda x: np.array(x.tolist()), [trainDF['feat'], trainDF[ratingCol], testDF['feat'], testDF[ratingCol]])

    def getVocabSize(self):
        return len(self.tokenizer.word_index)

    def getMaxSenLen(self):
        return self.maxSenLen

    def getTokenizerWordIndex(self):
        return self.tokenizer.word_index