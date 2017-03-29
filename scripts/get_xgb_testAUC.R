## function to build xgboost model and return k-fold test AUC
get_xgb_testAUC = function(max_depth, gamma, colsample_bytree,
                           min_child_weight, subsample, scale_pos_weight){
  param = list(objective = "binary:logistic", 
               booster = "gbtree",
               eval_metric = "auc",
               eta = init_eta,
               max_depth = max_depth,
               gamma = gamma,
               colsample_bytree = colsample_bytree,
               min_child_weight = min_child_weight,
               subsample = subsample,
               scale_pos_weight = scale_pos_weight
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
                 nthread = nthread,
                 verbose = FALSE,
                 set.seed(myseed)
  )
  
  ## k fold cross validation to estimate kfold AUC 
  ## with post-model feature engineering
  kfoldAUC = matrix(NA, nrow = kFolds, ncol = 1)
  for (i in 1:kFolds){
    left.out.data = dtrain[testFoldInd[[i]],]
    left.in.data = dtrain[trainFoldInd[[i]],]
    # train xgboost
    set.seed(myseed)
    xgbTrainModel = xgb.train(
      params = param,
      data = left.in.data,
      nrounds = xgbCV$best_ntreelimit,
      nthread = nthread,
      set.seed(myseed)
    )
    # make predictions on left out data with xgbmodel
    train_pred = predict(xgbTrainModel,left.out.data)
    # post-model feature engineering
    train_pred_post = post_model_feature_eng(modelTrainData[testFoldInd[[i]],],
                                             train_pred)
    # calculate test auc
    kfoldAUC[i,1] = colAUC(train_pred_post, modelTrainData$y[testFoldInd[[i]]])
  }
  # get the mean
  meanAUC = apply(kfoldAUC, MARGIN = 2, FUN = mean)
  print(c("meanAUC",meanAUC,"nrounds",xgbCV$best_ntreelimit))
  
  return(list(Score = meanAUC, Pred = 0))
  
}

## This is just a repeat of the function above with eta also as a parameter
## and early stop set to a new final value
## function to build xgboost model and return k-fold test AUC and nrounds used
get_xgb_testAUC_eta = function(max_depth, gamma, colsample_bytree,
                               min_child_weight, subsample, scale_pos_weight, eta){
  param = list(objective = "binary:logistic", 
               booster = "gbtree",
               eval_metric = "auc",
               eta = eta,
               max_depth = max_depth,
               gamma = gamma,
               colsample_bytree = colsample_bytree,
               min_child_weight = min_child_weight,
               subsample = subsample,
               scale_pos_weight = scale_pos_weight
  )
  # xgbCV to find optimal nround to stop at
  set.seed(myseed)
  xgbCV = xgb.cv(params = param,
                 data = dtrain,
                 nrounds = init_nrounds,
                 folds = testFoldInd,
                 early_stopping_rounds = final_early,
                 maximize = TRUE,
                 print_every_n = 1,
                 nthread = nthread,
                 verbose = FALSE,
                 set.seed(myseed)
  )
  
  ## k fold cross validation to estimate kfold AUC 
  ## with post-model feature engineering
  kfoldAUC = matrix(NA, nrow = kFolds, ncol = 1)
  for (i in 1:kFolds){
    left.out.data = dtrain[testFoldInd[[i]],]
    left.in.data = dtrain[trainFoldInd[[i]],]
    # train xgboost
    set.seed(myseed)
    xgbTrainModel = xgb.train(
      params = param,
      data = left.in.data,
      nrounds = xgbCV$best_ntreelimit,
      nthread = nthread,
      set.seed(myseed)
    )
    # make predictions on left out data with xgbmodel
    train_pred = predict(xgbTrainModel,left.out.data)
    # post-model feature engineering
    train_pred_post = post_model_feature_eng(modelTrainData[testFoldInd[[i]],],
                                             train_pred)
    # calculate test auc
    kfoldAUC[i,1] = colAUC(train_pred_post, modelTrainData$y[testFoldInd[[i]]])
  }
  # get the mean
  meanAUC = apply(kfoldAUC, MARGIN = 2, FUN = mean)
  print(c("meanAUC",meanAUC,"nrounds",xgbCV$best_ntreelimit))
  
  return(data.frame(Score = meanAUC, nrounds = xgbCV$best_ntreelimit))
  
}

## This is just a repeat of the first function with eta also as a parameter
## and nround as a parameter
## function to build final xgboost model and return final kaggle predictions
get_xgb_final_eta_nround = function(max_depth, gamma, colsample_bytree,
                                    min_child_weight, subsample, scale_pos_weight, eta, nrounds
){
  param = list(objective = "binary:logistic", 
               booster = "gbtree",
               eval_metric = "auc",
               eta = eta,
               max_depth = max_depth,
               gamma = gamma,
               colsample_bytree = colsample_bytree,
               min_child_weight = min_child_weight,
               subsample = subsample,
               scale_pos_weight = scale_pos_weight
  )
  # train xgboost
  set.seed(myseed)
  xgbTrainModel = xgb.train(
    params = param,
    data = dtrain,
    nrounds = nrounds,
    nthread = nthread,
    set.seed(myseed)
  )
  # make predictions on left out data with xgbmodel
  kaggle_pred = predict(xgbTrainModel,dtest)
  # post-model feature engineering
  kaggle_pred_post = post_model_feature_eng(modelTestData,kaggle_pred)
  
  return(kaggle_pred_post)
}