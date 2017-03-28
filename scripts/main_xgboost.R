## main file to run program
# github.com/jtran10/sds385_kaggle2

## notes:
# - see readme for program process description
# - program needs to run all at once to ensure reproducability by seed
# - init_eta, init_early, randLen, bayesLen for kaggle solution:
#      0.05, 75L, 150L, 50L
# - the above parameters are modified here for reasonable run time on laptop

## Clear workspace
rm(list=ls())

## load libraries
library(xgboost)
library(caTools)
library(caret)
library(rBayesianOptimization)
library(parallel)
library(tictoc)

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
init_eta = .05 
bounds = list(max_depth = c(1L, 20L),
              gamma = c(0.0,20.0),
              colsample_bytree = c(0.1, 1.0),
              min_child_weight = c(0L, 60L),
              subsample = c(0.1, 1.0),
              scale_pos_weight = c(0.1,10.0)
              )

## misc inputs to xgboost
init_early = 75L
nthread = detectCores()

## create random parameter search grid
randLen = 150L # number of search points in random grid
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
  tic()
  randGridAUC[i,1] = get_xgb_testAUC(random_grid$max_depth[i],
                                     random_grid$gamma[i],
                                     random_grid$colsample_bytree[i],
                                     random_grid$min_child_weight[i],
                                     random_grid$subsample[i],
                                     random_grid$scale_pos_weight[i])[[1]]
  toc()
}

## summarize random search results into data frame
random_grid_summary = data.frame(random_grid, Value = randGridAUC)
# use best N results as initial points for bayesian optimization
Nbest = 50L
ordered_by_value_rand_summary = 
  random_grid_summary[order(random_grid_summary$Value, decreasing = TRUE),]
rownames(ordered_by_value_rand_summary) <- NULL
initial_bayes_grid = ordered_by_value_rand_summary[1:Nbest,]


save.image(file="post_rand_search.RData")

## bayesian optimization of tuning parameters
bayesLen = 25L # number of search points for bayesian optimization
set.seed(myseed)
bayes_result = BayesianOptimization(
                  get_xgb_testAUC,
                  bounds = bounds,
                  init_grid_dt = initial_bayes_grid,
                  init_points = 0,
                  n_iter = bayesLen,
                  acq = "ucb",
                  kappa = 2.9,
                  eps = 0.0,
                  verbose = TRUE
)
save.image(file="post_rand_search_1.RData")



