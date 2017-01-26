# libraries
library('dplyr')
library('ggplot2')
library('GGally')
library('caret')
library('doParallel')
library('pROC')
library('mlbench')

# set random seed for reproducibility
set.seed(1337)

# load and clean data
data("BreastCancer")
mydata <- BreastCancer
mydata <- subset(mydata, select = -c(Id) ) # drop meaningless ID column
mydata <- dplyr::rename(mydata, class = Class)
mydata <- mydata[complete.cases(mydata),] # drop rows with missing values (note: you would probably use imputation of the mean / median instead in real scenarios)

# explore the data
#GGally::ggpairs(mydata, ggplot2::aes(colour = class)) + theme_bw()

# split data into training and test sets
intrain <- createDataPartition(mydata$class, p = 0.7 ,list=FALSE)
mydata_train <- mydata[intrain,]
mydata_test <- mydata[-intrain,]

# set formula: y = the 'class' column, x = all other columns
myformula <- as.formula("class ~ .")

ml.start.time <- Sys.time()

# start parallel computing clusters
cl <- makeCluster(detectCores())
registerDoParallel(cl)

## cross test settings
fitControl <- trainControl(
        method = "repeatedcv",
        number = 5,
        repeats = 5,
        savePredictions = T,
        classProbs = TRUE)

# neural network
mymodels_1 <- caret::train(myformula, metric = "Kappa", 
                              data = mydata_train,
                              method = "nnet",
                              trace = FALSE)
save(mymodels_1,file = "mymodels_nnet.Rdata")

# random forest
mymodels_2 <- caret::train(myformula, data= mydata_train, 
                               method="rf", silent = TRUE)
save(mymodels_2,file = "mymodels_rf.Rdata")

# J48 simple decision tree
mymodels_3 <- caret::train(myformula,
                       metric = "Kappa",
                       data = mydata_train,
                       method = "J48",
                       trControl = fitControl)
.jcache(mymodels_3$finalModel$classifier)
save(mymodels_3,file = "mymodels_j48.Rdata")

# confusion matrices for test data
confusionMatrix(predict(mymodels_1,mydata_test, type = "raw"),mydata_test$class)
confusionMatrix(predict(mymodels_2,mydata_test, type = "raw"),mydata_test$class)
confusionMatrix(predict(mymodels_3,mydata_test, type = "raw"),mydata_test$class)

myensembledata_train <- data.frame(
        mynnet = (predict(mymodels_1,mydata_train, type = "prob")),
        myrf = (predict(mymodels_2,mydata_train, type = "prob")),
        myJ48 = (predict(mymodels_3,mydata_train, type = "prob")),
        class = mydata_train$class)

mymodels_ensemble <- caret::train(myformula, data= myensembledata_train, 
                                        method = 'LogitBoost',
                                        metric = "Kappa",
                                        savePredictions = T)

save(mymodels_ensemble,file = "myensemblemodel.Rdata")

# end parallel computing clusters
stopCluster(cl)

ml.end.time <- Sys.time()

ml.run.time <- round(difftime(ml.end.time, ml.start.time , units = 'secs'),1)

myensembledata_test <- data.frame(
        mynnet = (predict(mymodels_1,mydata_test, type = "prob")),
        myrf = (predict(mymodels_2,mydata_test, type = "prob")),
        myJ48 = (predict(mymodels_3,mydata_test, type = "prob"))
        )

# generate ensemble predictions
myensembledata_test_ensemble_preds <- (predict(mymodels_ensemble,myensembledata_test, type = "prob"))

# rename ensemble columns
myensembledata_test_ensemble_preds <- dplyr::rename(myensembledata_test_ensemble_preds, ensemble.benign = benign, ensemble.malignant = malignant)

# add ensemble predictions to test data set predictions
myensembledata_test <- cbind(myensembledata_test,myensembledata_test_ensemble_preds)

# add the class
myensembledata_test$class <- mydata_test$class

# plot ROC curve of ensemble classifier
plot(roc(myensembledata_test$class , myensembledata_test$ensemble.malignant, direction="<"))

mycfm_model_1 <- confusionMatrix(predict(mymodels_1, mydata_test[,col(mydata_test)-1]),
                                 mydata_test$class,
                                 positive = "malignant")

mycfm_model_2 <- confusionMatrix(predict(mymodels_2, mydata_test[,col(mydata_test)-1]),
                                 mydata_test$class,
                                 positive = "malignant")

mycfm_model_3 <- confusionMatrix(predict(mymodels_3, mydata_test[,col(mydata_test)-1]),
                                 mydata_test$class,
                                 positive = "malignant")

mycfm_ensemble <- confusionMatrix(predict(mymodels_ensemble,
                                        myensembledata_test[,col(myensembledata_test)-1]),
                                        myensembledata_test$class,
                                        positive = "malignant")

print(mycfm_ensemble)

ggpairs(subset(myensembledata_test, select = c(mynnet.malignant, myrf.malignant, myJ48.malignant, ensemble.malignant, class) ),ggplot2::aes(colour = class)) +
        theme_bw()