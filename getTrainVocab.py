from DataReader import ReviewsReader

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

l = []
for word, i in reviews.getTokenizerWordIndex().items():
    l += [word]

l = set(l)
l = list(l)
l = sorted(l)
with open('trainVocab.txt', 'w', encoding='utf8') as f:
    for i in l:
        f.write(i + '\n')