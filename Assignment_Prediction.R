
#######################################################################
########## Coursera Pratical Machine Learning Course Project ##########
#######################################################################

### PROJECT DESCRIPTION: 
### Use data gathered from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.
### The goal of the project is to predict the manner (classe) in which people they exercise. 


library(caret)
library(pROC)
library(ggplot2)


# 1. Get Data from CSV
# The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.
train <- read.csv("C:\\Users\\Swan\\Desktop\\Codes\\pml-training.csv")
score <- read.csv("C:\\Users\\Swan\\Desktop\\Codes\\pml-testing.csv")


# 2. Split Train-Test 60:40 - using Caret
set.seed(695)
inTrain <- createDataPartition(y=train$classe, p=0.60, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]


# 3. Understand data
# # a. basic tools
# head(training,6)
# str(training)
# colnames(training)
# summary(training)
# ggplot(data.frame(training$classe), aes(x=training$classe)) +
#   geom_bar()
# 
# # b. Feature Plot - caret
# featurePlot(x=training[,c(20,42,52)], y=training$classe, plot="pairs")


# 4. Preprocess
# Remove target & problem ID
train_tv <- training$classe
training <- training[,-160]
test_tv <- testing$classe
testing <- testing[,-160]
score <- score[,-160]

# Choose variables to Remove
variables <- c('user_name', 'num_window', 'roll_belt', 'pitch_belt', 'yaw_belt', 'total_accel_belt',
               'gyros_belt_x', 'gyros_belt_y', 'gyros_belt_z', 'accel_belt_x', 'accel_belt_y', 'accel_belt_z',
               'magnet_belt_x', 'magnet_belt_y', 'magnet_belt_z', 'roll_arm', 'pitch_arm', 'yaw_arm',
               'total_accel_arm', 'gyros_arm_x', 'gyros_arm_y', 'gyros_arm_z', 'accel_arm_x', 'accel_arm_y',
               'accel_arm_z', 'magnet_arm_x', 'magnet_arm_y', 'magnet_arm_z', 'roll_dumbbell', 'pitch_dumbbell',
               'yaw_dumbbell', 'total_accel_dumbbell', 'gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z',
               'accel_dumbbell_x', 'accel_dumbbell_y', 'accel_dumbbell_z', 'magnet_dumbbell_x', 'magnet_dumbbell_y',
               'magnet_dumbbell_z', 'roll_forearm', 'pitch_forearm', 'yaw_forearm', 'total_accel_forearm',
               'gyros_forearm_x', 'gyros_forearm_y', 'gyros_forearm_z', 'accel_forearm_x', 'accel_forearm_y',
               'accel_forearm_z', 'magnet_forearm_x', 'magnet_forearm_y', 'magnet_forearm_z')
train_df <- training[,variables]
test_df <- testing[,variables]
score_df <- score[,variables]

# a. Zero Variance
nzv <- nearZeroVar(score, saveMetrics=TRUE)
nzv[nzv$nzv,][1:10,]
nzv <- nearZeroVar(score)
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


# 5. Model training
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
