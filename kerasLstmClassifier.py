from DataReader import ReviewsReader

r = ReviewsReader()
xt, yt, xTest, yTest = r.readTrainTest()
print(xt)