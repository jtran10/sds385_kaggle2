## main file to run entire program

## Clear workspace
rm(list=ls())

## load libraries
library(xgboost)
library(Matrix)
library(pROC)

## set working directory
this.dir = dirname(parent.frame(2)$ofile)
setwd(this.dir)

## load the data and do some pre-model feature engineering
source('data_loading_feature_engineering.R')

## set seed
myseed = 555
set.seed(myseed)

## 
