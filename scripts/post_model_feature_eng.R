## post manipulation of model predictions based on feature engineering

post_model_feature_eng = function(predData, pred){
  post_pred = pred
  
  # feature 1
  index = which(predData$x4 > 60000 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  # feature 2
  index = which(predData$x4 > 2950 &
                  predData$x3 == 0 &
                  predData$x7 == 0 &
                  predData$x9 == 0 &
                  predData$x8 == 0 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  # feature 3
  index = which(predData$x4> 8000 &
                  predData$x8 == 1 &
                  predData$x3 == 0 &
                  predData$x7 == 0 &
                  predData$x9 == 0 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  # feature 4
  index = which(predData$x4 > 80 &
                  predData$x2 > 80 &
                  predData$x3 == 0 &
                  predData$x7 == 0 &
                  predData$x8 == 0 &
                  predData$x9 == 0 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  # feature 5
  index = which(predData$x4 > 4200 &
                  predData$x2 >= 70 &
                  predData$x2 <= 80 &
                  predData$x3 == 0 &
                  predData$x7 == 0 &
                  predData$x9 == 0 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  # feature 6
  index = which(predData$x4 > 8000 &
                  predData$x2 >= 60 &
                  predData$x2 < 70 &
                  predData$x3 == 0 &
                  predData$x7 == 0 &
                  predData$x9 == 0 &
                  pred > 0.000001)
  post_pred[index] = 0.000001
  
  return(post_pred)
}

