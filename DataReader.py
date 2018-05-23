import pandas
from keras.preprocessing.text import Tokenizer

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

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.reviews['review'].tolist())
        self.reviews['feat'] = self.reviews['review'].apply(lambda x: self.tokenizer.texts_to_sequences([x])[0])
        self.reviews['2classRating'] = self.reviews['rating'].apply(lambda x: 1 if x > 3 else 0)
    
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
        
        ratingCol = '2classRating' if twoClass else 'rating'
        return map(lambda x: x.tolist(), [trainDF['feat'], trainDF[ratingCol], testDF['feat'], testDF[ratingCol]])

    def getVocabSize(self):
        return len(self.tokenizer.word_index)

    def getMaxSenLen(self):
        return self.reviews['feat'].apply(lambda x: len(x)).max()
