### Loan default prediction modeling project for SDS385. 

The goal is to predict if a person will default or not. The data set is moderately imbalanced (only ~6% default). The metric score is given as AUC-ROC. 

The approach that is taken here is to use feature engineering, xgboost, and blackbox optimization of tuning parameters. The goal is to try to maximize the potential of an xgboost classification model. Feature engineering is done prior to xgboost and afterwards. Tuning parameters are found with a random search + bayesian optimization process.