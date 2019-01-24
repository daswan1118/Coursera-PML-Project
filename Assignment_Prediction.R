
#######################################################################
########## Coursera Pratical Machine Learning Course Project ##########
#######################################################################

### PROJECT DESCRIPTION: 
### Use data gathered from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
### The goal of the project is to predict the manner (classe) in which people they exercise. 
### The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

library(caret)
library(pROC)
library(ggplot2)


## 1. Get Data from CSV & Split Data

# a. Get data
train <- read.csv("pml-training.csv")
score <- read.csv("pml-testing.csv")

# b. Split data into 60:40
set.seed(695)
inTrain <- createDataPartition(y=train$classe, p=0.60, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]


## 2. Understand Data & Feature Selection

# a. Basic tools
head(training,6)
str(training)
colnames(training)
summary(training)
ggplot(data.frame(training$classe), aes(x=training$classe)) +
  geom_bar()

# b. Feature Plot - caret
featurePlot(x=training[,c(2,8,160)], y=training$classe, plot="pairs")

# c. Remove target & problem ID
train_tv <- training$classe
training <- training[,-160]
test_tv <- testing$classe
testing <- testing[,-160]
score <- score[,-160]

# d. Remove columns with all NA in score
training <- training[, colSums(is.na(score)) != nrow(score)]
testing <- testing[, colSums(is.na(score)) != nrow(score)]
score <- score[, colSums(is.na(score)) != nrow(score)]
colnames(training)

# e. Choose variables to remove
variables <- c('X', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp')
train_df <- training[,!(names(training) %in% variables)]
test_df <- testing[,!(names(testing) %in% variables)]
score_df <- score[,!(names(score) %in% variables)]


## 3. Preprocess

# a. Zero Variance
nzv <- nearZeroVar(train_df, saveMetrics=TRUE)
nzv[nzv$nzv,][1:10,]
nzv <- nearZeroVar(train_df)
train_df <- train_df[, -nzv]
test_df <- test_df[, -nzv]
score_df <- score_df[, -nzv]

# b. convert factor variables to indicator (1,0) variables  
dummies <- dummyVars(~., data=train_df)
train_df <- as.data.frame(predict(dummies, newdata=train_df))
test_df <- as.data.frame(predict(dummies, newdata=test_df))
score_df <- as.data.frame(predict(dummies, newdata=score_df))

# c. Correlated Predictors
descrCor <-  cor(train_df, use="pairwise.complete.obs")
descrCor[is.na(descrCor)] <- 0
highlyCorDescr <- findCorrelation(descrCor, cutoff = .90)
highlyCorDescr
train_df <- train_df[,-highlyCorDescr]
test_df <- test_df[,-highlyCorDescr]
score_df <- score_df[,-highlyCorDescr]

# d. Standardizing - Imputing Data
preObj <- preProcess(train_df, method = "medianImpute")
train_df <- predict(preObj,train_df)
test_df <- predict(preObj,test_df)
score_df <- predict(preObj,score_df)


## 4. Model training

# a. Fit Random Forest model
fitcontrol <- fitControl <- trainControl(method = 'cv', number = 3, verbose = TRUE)
tunegrid <- expand.grid(.mtry=20)
set.seed(695)
rrfFit <- train(train_df, train_tv, method = "rf", trControl = fitControl,
                ntree = 150, tuneGrid=tunegrid, importance = TRUE)
test_imp_rf <- varImp(rrfFit, scale = FALSE); test_imp_rf[[1]]["A"]; plot(test_imp_rf)

# b. Predict on test set
test_pred <- predict(rrfFit, test_df)
confusionMatrix(test_pred, test_tv)

# c. Predict on score set
score$classe_pred <- predict(rrfFit, score_df)
score[,c("X","classe_pred")]
