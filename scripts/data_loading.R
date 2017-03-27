# Loading training and test data
this.dir = dirname(parent.frame(2)$ofile)
setwd(this.dir)
trainData = read.csv("../data/train.csv", header = T)
testData = read.csv("../data/test.csv", header = T)

# Remove ID column
trainData = subset(trainData, select = -Id)
testData = subset(testData, select = -Id)

# Remove data point with mislabelled age
index = which(trainData$x2 == 0)
trainAgeData = trainData[-index,]

# Create binary predictor for 9698 data
index = which(trainAgeData$x3 %in% c(96,98))
train9698tempData = trainAgeData
train9698tempData[index, c("x3","x7","x9")] = 0
b9698 = data.frame(b9698 = matrix(0, dim(train9698tempData)[1], 1))
b9698$b9698[index] = 1
b9698$b9698 = factor(b9698$b9698, labels = c("n","y"))
train9698Data = as.data.frame(c(train9698tempData,b9698))

# 9698 column for test data
index = which(testData$x3 %in% c(96,98))
test9698tempData = testData
test9698tempData[index, c("x3","x7","x9")] = 0
testb9698 = data.frame(b9698 = matrix(0, dim(test9698tempData)[1], 1))
testb9698$b9698[index] = 1
testb9698$b9698 = factor(testb9698$b9698, labels = c("n","y"))
test9698Data = as.data.frame(c(test9698tempData,testb9698))

## clean up
rm(train9698tempData)
rm(test9698tempData)
rm(b9698)
rm(testb9698)
rm(index)
rm(this.dir)













