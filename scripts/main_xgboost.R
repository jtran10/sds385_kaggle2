## main file to run entire program
# github.com/jtran10/sds385_kaggle2

## Clear workspace
rm(list=ls())

## load libraries
library(xgboost)
library(Matrix)
library(caTools)
library(caret)

## set working directory
this.dir = dirname(parent.frame(2)$ofile)
setwd(this.dir)

## load the data and do some pre-model feature engineering
source('data_loading_feature_engineering.R')

## set seed
myseed = 555L
set.seed(myseed)

# select data set to use
modelTrainData = train9698Data
modelTestData = test9698Data

## create k-fold lists (test indices)
kFolds = 10
set.seed(myseed)
testFoldInd = createFolds(modelTrainData$y, kFolds, returnTrain = FALSE)
set.seed(myseed)
trainFoldInd = createFolds(modelTrainData$y, kFolds, returnTrain = TRUE)

## input initial constant tuning params and bounds
init_nrounds = 50000L # basically infinite
init_eta = .01 
bounds = list(max_depth = c(1L, 20L),
              gamma = c(0.0,20.0),
              colsample_bytree = c(0.1, 1.0),
              min_child_weight = c(0L, 60L),
              subsample = c(0.1, 1.0),
              scale_pos_weight = c(0.1,8.0)
              )

## misc inputs to xgboost
init_early = 3L

## create random parameter search grid
randLen = 4 # numer of search points in random grid
source('rand_search_grid.R')
set.seed(myseed)
random_grid = create_random_grid(randLen, bounds)

## manipulate data to be usable for xgboost
xgbTrainData = model.matrix(y ~ ., data = modelTrainData)[,-1]
xgbTestData = model.matrix(~ ., data = modelTestData)[,-1]
train.y = ifelse(modelTrainData$y == "y", 1, 0)
dtrain = xgb.DMatrix(xgbTrainData, label = train.y)

##




