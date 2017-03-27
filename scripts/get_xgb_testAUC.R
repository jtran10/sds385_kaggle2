## function to build xgboost model and return k-fold test AUC

param = list(objective = "binary:logistic", 
             booster = "gbtree",
             eval_metric = "auc",
             eta = init_eta,
             max_depth = random_grid$max_depth[1],
             gamma = random_grid$gamma[1],
             colsample_bytree = random_grid$colsample_bytree[1],
             min_child_weight = random_grid$min_child_weight[1],
             subsample = random_grid$subsample[1],
             scale_pos_weight = random_grid$scale_pos_weight[1]
)
# xgbCV to find optimal nround to stop at
set.seed(myseed)
xgbCV = xgb.cv(params = param,
               data = dtrain,
               nrounds = init_nrounds,
               folds = testFoldInd,
               early_stopping_rounds = init_early,
               maximize = TRUE,
               print_every_n = 1,
               set.seed(myseed)
)
# k fold cross validation to estimate kfold AUC 
# with post-model feature engineering
kfoldAUC = matrix(NA, nrow = kFolds, ncol = 1)
tic()
for (i in 1:kFolds){
  left.out.data = dtrain[testFoldInd[[i]],]
  left.in.data = dtrain[trainFoldInd[[i]],]
  # train xgboost
  set.seed(myseed)
  xgbTrainModel = xgb.train(
    params = param,
    data = left.in.data,
    nrounds = xgbCV$best_ntreelimit,
    set.seed(myseed)
  )
  # post-model feature engineering
  
  # make predictions on left out data
  train_pred = predict(xgbTrainModel,left.out.data)
  # calculate test auc
  kfoldAUC[i,1] = colAUC(train_pred, modelTrainData$y[testFoldInd[[i]]])
}
toc()
meanAUC = apply(kfoldAUC, MARGIN = 2, FUN = mean)



