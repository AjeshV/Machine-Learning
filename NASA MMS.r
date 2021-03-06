---
title: "Final Source of prediction models for NASA Magnetospheric Multiscale Mission"
output:
  html_document:
    df_print: paged
  html_notebook: default
  pdf_document: default
---

```{r}
library(tidyverse)
library(lubridate)
library(magrittr)
library(fuzzyjoin)
# print milliseconds
options(digits.secs=3)
# To make sure that X1 is a correct row index
mms <- read_csv("mms_20151016.csv") %>% mutate(X1 = X1 + 1)
sitl <- read_csv("sitl_20151016.csv")
mms.id <- mms %>% transmute(Start = Time, End = Time, X1 = X1)
sitl.id <- sitl %>% select(c(Start,End,Priority))
joined <- interval_inner_join(sitl.id, mms.id) %>% group_by(X1) %>%
                summarize(Priority = max(Priority))
mms.target <- mms 
mms.target$Selected <- FALSE
mms.target$Selected[joined$X1] <- TRUE
mms.target$Priority <- 0
mms.target$Priority[joined$X1] <- joined$Priority
# second
mms1 <- read_csv("mms_20151204.csv") %>% mutate(X1 = X1 + 1)
sitl1 <- read_csv("sitl_20151204.csv")
mms1.id <- mms1 %>% transmute(Start = Time, End = Time, X1 = X1)
sitl1.id <- sitl1 %>% select(c(Start,End,Priority))
joined <- interval_inner_join(sitl1.id, mms1.id) %>% group_by(X1) %>%
                summarize(Priority = max(Priority))
mms1.target <- mms1 
mms1.target$Selected <- FALSE
mms1.target$Selected[joined$X1] <- TRUE
mms1.target$Priority <- 0
mms1.target$Priority[joined$X1] <- joined$Priority
# third test
mmstest.target <- read_csv("mms_20161022.csv") %>% mutate(X1 = X1 + 1)
#merge(mms,mms1)
mmsfull.target <- full_join(mms.target,mms1.target)
```

```{r}
#write.csv(mms.target, file = "mms.csv",row.names=FALSE)
#write.csv(mms1.target, file = "mms1.csv",row.names=FALSE)
mms1.target = read.csv("mms1.csv")
mms.target = read.csv("mms.csv")
```

bagging
```{r}
train = mms1.target
test = mms.target
train = train[,c(19,18,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)]
test = test[,c(19,18,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)]
train$Selected=as.factor(as.numeric(train$Selected))
test$Selected=as.factor(as.numeric(test$Selected))
train = train[,c(-2,-3,-4,-5,-6,-9,-15)]
test = test[,c(-2,-3,-4,-5,-6,-9,-15)]
#splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
#Fitting Random Forest to the Training Set
#install.packages('randomForest')
library(randomForest)
classifier=randomForest(y=train[,1],x=train[,-1],data =,ntree =500,mtry=12)
y_pred1=predict(classifier,newdata=test,type='class')#take away 3rd elem
cat("\n Bagging forest test classification error", mean(y_pred1 != test$Selected ))
```

random forest
```{r}
train = mms1.target
test = mms.target
train = train[,c(19,18,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)]
test = test[,c(19,18,20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)]
train$Selected=as.factor(as.numeric(train$Selected))
test$Selected=as.factor(as.numeric(test$Selected))
train = train[,c(-2,-3,-4,-5,-6,-9,-15)]
test = test[,c(-2,-3,-4,-5,-6,-9,-15)]
#splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
#Fitting Random Forest to the Training Set
#install.packages('randomForest')
library(randomForest)
classifier=randomForest(y=train[,1],x=train[,-1],data =,ntree =500)
y_pred1=predict(classifier,newdata=test,type='class')#take away 3rd elem
cat("\n random forest test classification error", mean(y_pred1 != test$Selected ))
```

knn
```{r}
train1 = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test1 = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train1 = train1[,-c(2, 3, 4, 5, 6)]
test1 = test1[,-c(2, 3, 4, 5, 6)]
train = train1[,-c(2,3,4,5,7,8,13,14)]
test = test1[,-c(2,3,4,5,7,8,13,14)]
library(gbm)
#train1$Selected = as.factor((train1$Selected))
#test1$Selected = as.factor((test1$Selected))
library(class)
#first k = 1
knn <- knn(train,test,train$Selected,k=300)
table(knn,test$Selected)
cat( "\nError "," ->",mean(knn!=test$Selected))
```

SVM radial (PCA)
```{r}
library(caret)
library(e1071)
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
train$Selected = as.factor((train$Selected))
test$Selected = as.factor((test$Selected))
dim = 2
  pca = preProcess(x = train,method = 'pca',pcaComp = dim)
  i = 1 + dim
  set = c(c(2:i), 1)
  train = predict(pca, train)
  train = train[set]
  test = predict(pca, test) #change your test set into a new one
  test = test[set] 
  svmr = svm(Selected ~ ., data = train, kernel = "radial")
  svmrpredt = predict(svmr, newdata = test)
    cat("\nradial test classification error " ,mean(svmrpredt != test$Selected))
```

SVM radial
```{r}
library(caret)
library(e1071)
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
train$Selected = as.factor((train$Selected))
test$Selected = as.factor((test$Selected))
  
svmr = svm(Selected ~ ., data = train, kernel = "radial")
svmrpredt = predict(svmr, newdata = test)
cat("\nradial test classification error " ,mean(svmrpredt != test$Selected))
```

boosting
```{r}
library(gbm)
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
train$Selected=as.numeric(train$Selected)
test$Selected=as.numeric(test$Selected)
library(caTools)
gbm.model = gbm(Selected ~ ., data=train, distribution = 'bernoulli', n.trees=9000,interaction.depth = 20)
y_pred= predict(object = gbm.model,
                              newdata = test,
                              n.trees = 5000,
                              type = "response")
y_pred[which(y_pred>=0.839)]=1
y_pred[which(y_pred<0.839)]=0
table(y_pred,test$Selected)
cat("\n boosting test classification error" ,mean(y_pred != test$Selected))
```

lda
```{r}
library(MASS)
library(caret)
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
ldatrain = lda(Selected ~ ., data=train)
ldapredict = predict(ldatrain,test)
table(ldapredict$class,test$Selected)
cat("\nLDA classification error" ,1- mean(ldapredict$class ==test$Selected))
```

qda
```{r}
library(MASS)
library(caret)
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
ldatrain = qda(Selected ~ ., data=train)
ldapredict = predict(ldatrain,test)
table(ldapredict$class,test$Selected)
cat("\nQDA classification error" ,1- mean(ldapredict$class ==test$Selected))
```

logistic regression
```{r}
train = mms1.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mms.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
bostonlg <- glm(Selected ~ ., data=train, family=binomial)
bostonlgprob=predict(bostonlg, test,type="response")
bostonlgpredict=rep(0,nrow(test))
bostonlgpredict[bostonlgprob > 0.50]=1
table(bostonlgpredict,test$Selected)
cat("\nlogistic classification error" ,1- mean(bostonlgpredict==test$Selected))
```

SVM radial (PCA) // full predict
```{r}
library(caret)
library(e1071)
train = mmsfull.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mmstest.target[, c(18, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
train = train[,-c(2, 3, 4, 5, 6)]
test = test[,-c(1,2, 3, 4)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
test = test[,-c(1, 2, 3, 4, 6, 7,12,13)]
train$Selected = as.factor((train$Selected))
dim = 2
  pca = preProcess(x = train,method = 'pca',pcaComp = dim)
  i = 1 + dim
  set = c(c(2:i), 1)
  train = predict(pca, train)
  train = train[set]
  test = predict(pca, test) #change your test set into a new one
  #test = test[set]
  svmr = svm(Selected ~ ., data = train, kernel = "radial")
  svmrpredt = predict(svmr, newdata = test)  
  mitl_filename <- "predict/SVMmitl_20161022.csv"
  output <- data_frame(ObservationId=mmstest.target$X1,Selected=c(svmrpredt))
  write_csv(output, mitl_filename)
   # cat("\nradial test classification error " ,mean(svmrpredt != test$Selected))
```

boosting
```{r}
library(gbm)
train = mmsfull.target[, c(19, 18, 20, 8, 9, 1, 11, 12, 13, 10, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17)]
test = mmstest.target
train = train[,-c(2, 3, 4, 5, 6)]
train = train[,-c(2, 3, 4, 5, 7, 8, 13, 14)]
train$Selected=as.numeric(train$Selected)
library(caTools)
#gbm.model = gbm(Selected ~ ., data=train, distribution = 'bernoulli', n.trees=5000,interaction.depth = 15)
y_pred_boosting= predict(object = gbm.model,
                              newdata = test,
                              n.trees = 5000,
                              type = "response")
y_pred_boosting[which(y_pred_boosting>=0.839)]=1
y_pred_boosting[which(y_pred_boosting<0.839)]=0
mitl_filename <- "predict/BOOSTINGmitl_20161022.csv"
output <- data_frame(ObservationId=mmstest.target$X1,Selected=c(y_pred_boosting))
write_csv(output, mitl_filename)
```