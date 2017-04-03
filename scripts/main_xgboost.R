## main file to run program
# github.com/jtran10/sds385_kaggle2

## notes:
# - see readme for program process description
# - program needs to run all at once to ensure reproducability by seed
# - kFolds, init_eta, init_early, randLen, Nbest, bayesLen, 
#      bayesLen2, eta, final_early for kaggle solution:
#      10L, 0.05, 75L, 150L, 50L, 25L, 50L, c(.001,.005,.01,.05,.1), 40L
# - the above parameters are modified here for reasonable run time on a laptop

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
kFolds = 3L
set.seed(myseed)
testFoldInd = createFolds(modelTrainData$y, kFolds, returnTrain = FALSE)
set.seed(myseed)
trainFoldInd = createFolds(modelTrainData$y, kFolds, returnTrain = TRUE)

## input initial constant tuning params and bounds
init_nrounds = 50000L # basically infinite
init_eta = .3 
bounds = list(max_depth = c(1L, 20L),
              gamma = c(0.0,20.0),
              colsample_bytree = c(0.1, 1.0),
              min_child_weight = c(0L, 60L),
              subsample = c(0.1, 1.0),
              scale_pos_weight = c(0.1,10.0)
)

## misc inputs to xgboost
init_early = 5L
nthread = detectCores()

## create random parameter search grid
randLen = 10L # number of search points in random grid
source('rand_search_grid.R')
set.seed(myseed)
random_grid = create_random_grid(randLen, bounds)

## manipulate data to be usable for xgboost
xgbTrainData = model.matrix(y ~ ., data = modelTrainData)[,-1]
xgbTestData = model.matrix(~ ., data = modelTestData)[,-1]
train.y = ifelse(modelTrainData$y == "y", 1, 0)
dtrain = xgb.DMatrix(xgbTrainData, label = train.y)
dtest = xgb.DMatrix(xgbTestData)

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
Nbest = 3L
ordered_by_value_rand_summary = 
  random_grid_summary[order(random_grid_summary$Value, decreasing = TRUE),]
rownames(ordered_by_value_rand_summary) <- NULL
initial_bayes_grid = ordered_by_value_rand_summary[1:Nbest,]
# save workspace in case of crash
save.image(file="post_rand_search.RData")

## bayesian optimization of tuning parameters
bayesLen = 2L # number of search points for bayesian optimization
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
# save workspace in case of crash
save.image(file="post_bayes_search_1.RData")

## bayesian optimization with small kappa
initial_bayes_grid2 = bayes_result$History[,-1]
bayesLen2 = 2L
set.seed(myseed)
bayes_result2 = BayesianOptimization(
  get_xgb_testAUC,
  bounds = bounds,
  init_grid_dt = initial_bayes_grid2,
  init_points = 0,
  n_iter = bayesLen2,
  acq = "ucb",
  kappa = 0.15,
  eps = 0.0,
  verbose = TRUE
)
# save workspace in case of crash
save.image(file="post_bayes_search_2.RData")

## Extract best model after bayesian optimization + random search
BayesResult = bayes_result2$History[,-1]
ordered_by_value_bayes = 
  BayesResult[order(BayesResult$Value, decreasing = TRUE),]
best_bayes_grid = ordered_by_value_bayes[1,]

## create eta grid with best_bayes_grid
eta = as.matrix(c(.1,.3,.5))
best_bayes_repeat_grid = best_bayes_grid[rep(seq_len(nrow(best_bayes_grid)), 
                                             each=length(eta)),]
best_bayes_eta_grid = data.frame(best_bayes_repeat_grid, eta = eta)
rownames(best_bayes_eta_grid) = NULL

## search for the best model varying eta
final_early = 5L
etaGridAUC = data.frame(matrix(NA,dim(best_bayes_repeat_grid)[1],2L))
colnames(etaGridAUC) = c("Score","nrounds")
for (i in 1:dim(best_bayes_repeat_grid)[1]){
  print(c("etaIter",i))
  tic()
  etaGridAUC[i,] = get_xgb_testAUC_eta(best_bayes_eta_grid$max_depth[i],
                                       best_bayes_eta_grid$gamma[i],
                                       best_bayes_eta_grid$colsample_bytree[i],
                                       best_bayes_eta_grid$min_child_weight[i],
                                       best_bayes_eta_grid$subsample[i],
                                       best_bayes_eta_grid$scale_pos_weight[i],
                                       best_bayes_eta_grid$eta[i])
  toc()
}

## train the best model and get predictions on kaggle test.csv
best_model_index = which.max(etaGridAUC$Score)
kaggle_final_pred = get_xgb_final_eta_nround(
  best_bayes_eta_grid$max_depth[best_model_index],
  best_bayes_eta_grid$gamma[best_model_index],
  best_bayes_eta_grid$colsample_bytree[best_model_index],
  best_bayes_eta_grid$min_child_weight[best_model_index],
  best_bayes_eta_grid$subsample[best_model_index],
  best_bayes_eta_grid$scale_pos_weight[best_model_index],
  best_bayes_eta_grid$eta[best_model_index],
  etaGridAUC$nrounds[best_model_index]
)
# save workspace in case of crash
save.image(file = "post_final_eta_search_final_pred.Rdata")

## output submission file
kaggle_pred_df = data.frame(Id = 1:75000, y = kaggle_final_pred)
write.csv(kaggle_pred_df , file = "final.csv" , row.names = FALSE)

## final model
print("final model parameters")
print(cbind(best_bayes_eta_grid[best_model_index,-7], 
      nrounds=etaGridAUC$nrounds[best_model_index]),
      CVAUC = etaGridAUC$Score[best_model_index])

