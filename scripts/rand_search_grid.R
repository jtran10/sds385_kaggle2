## function to create return random search grid
create_random_grid <- function(randLen, bounds ){
  set.seed(myseed)
random_grid = data.frame(
  max_depth = sample(bounds$max_depth[1]:bounds$max_depth[2], 
                     replace = TRUE, size = randLen),
  gamma = runif(randLen, min = bounds$gamma[1], max = bounds$gamma[2]),
  colsample_bytree = runif(randLen, min = bounds$colsample_bytree[1], 
                           max = bounds$colsample_bytree[2]),
  min_child_weight = sample(bounds$min_child_weight[1]:bounds$min_child_weight[2], 
                            replace = TRUE, size = randLen),
  subsample = runif(randLen, min = bounds$subsample[1], max = bounds$subsample[2]),
  scale_pos_weight = runif(randLen, min = bounds$scale_pos_weight[1], 
                           max = bounds$scale_pos_weight[2])
  )
return(random_grid)
}