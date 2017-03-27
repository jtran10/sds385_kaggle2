## main file to run program
# github.com/jtran10/sds385_kaggle2

## notes:
# - program needs to run all at once to ensure reproducability by seed
# 

## Clear workspace
rm(list=ls())

## load libraries
library(xgboost)
library(caTools)
library(caret)
library(rBayesianOptimization)
library(parallel)

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
kFolds = 10L
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
nthread = detectCores()


## create random parameter search grid
randLen = 4L # number of search points in random grid
source('rand_search_grid.R')
set.seed(myseed)
random_grid = create_random_grid(randLen, bounds)

## manipulate data to be usable for xgboost
xgbTrainData = model.matrix(y ~ ., data = modelTrainData)[,-1]
xgbTestData = model.matrix(~ ., data = modelTestData)[,-1]
train.y = ifelse(modelTrainData$y == "y", 1, 0)
dtrain = xgb.DMatrix(xgbTrainData, label = train.y)

## get all kfold test AUC for random grid
source('post_model_feature_eng.R')
source('get_xgb_testAUC.R')
randGridAUC = matrix(NA,randLen,1L)
for (i in 1:randLen){
  print(c("randIter",i))
  randGridAUC[i,1] = get_xgb_testAUC(random_grid$max_depth[i],
                                     random_grid$gamma[i],
                                     random_grid$colsample_bytree[i],
                                     random_grid$min_child_weight[i],
                                     random_grid$subsample[i],
                                     random_grid$scale_pos_weight[i])[[1]]
}

## summarize random search results into data frame
initial_bayes_grid = data.frame(random_grid, Value = randGridAUC)

## bayesian optimization of tuning parameters
bayesLen = 4L # number of search points for bayesian optimization
bayes_result = BayesianOptimization(
                  get_xgb_testAUC,
                  bounds = bounds,
                  init_grid_dt = initial_bayes_grid,
                  init_points = 0,
                  n_iter = 4,
                  acq = "ucb",
                  kappa = 4.0,
                  eps = 0.0,
                  verbose = TRUE
)


save.image(file="test.RData")



