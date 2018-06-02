import keras
from DataReader import ReviewsReader

reviews = ReviewsReader()
xTrain, yTrain, xTest, yTest = reviews.readTrainTest(twoClass=True, balanced=False)

### pick subset from test as validation
valReviewList, xVal, yVal = reviews.readValidationData()

model = keras.models.load_model('./wordModels/modelChk.09-0.90.hdf5')
print(model.evaluate(xVal, yVal, batch_size=128))

predictions = model.predict(xVal)
predictions = [1 if i >= 0.5 else 0 for i in predictions]

with open('errors.txt', 'w', encoding='utf8') as f:
    f.write('Correct\tPredict\tReview\n')
    for i in range(len(valReviewList)):
        if predictions[i] != yVal[i]:
            f.write('\t'.join([str(yVal[i]), str(predictions[i]), valReviewList[i]]))
            f.write('\n')

with open('correct.txt', 'w', encoding='utf8') as f:
    f.write('Correct\tPredict\tReview\n')
    for i in range(len(valReviewList)):
        if predictions[i] == yVal[i]:
            f.write('\t'.join([str(yVal[i]), str(predictions[i]), valReviewList[i]]))
            f.write('\n')
