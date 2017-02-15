# libraries
library('dplyr')
library('ggplot2')
library('GGally')
library('caret')
library('doParallel')
library('pROC')
library('mlbench')
library('vtreat')

# set random seed for reproducibility
set.seed(1337)

# load and clean data
data("BreastCancer")
mydata <- BreastCancer
mydata <- subset(mydata, select = -c(Id) ) # drop meaningless ID column
mydata <- dplyr::rename(mydata, class = Class)

# explore the data
#GGally::ggpairs(mydata, ggplot2::aes(colour = class)) + theme_bw()

# split data into training and test sets
intrain <- createDataPartition(mydata$class, p = 0.7 ,list=FALSE)
mydata_train <- mydata[intrain,]
mydata_test <- mydata[-intrain,]

yName <- 'class'
yTarget <- 'malignant'
varNames <- (setdiff(colnames(mydata_train),yName))

# design a preprocessing plan
train_treatmentsC <- designTreatmentsC(dframe = mydata_train,
                                               varlist = varNames,
                                               outcomename = yName,
                                               outcometarget = yTarget)

mydata_train_treated <- vtreat::prepare(train_treatmentsC, 
                                             mydata_train, 
                                             pruneSig = 0.05)

mydata_test_treated <- vtreat::prepare(train_treatmentsC, 
                                        mydata_test, 
                                        pruneSig = 0.05)


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
                              data = mydata_train_treated,
                              method = "nnet",
                              trace = FALSE)
save(mymodels_1,file = "mymodels_nnet.Rdata")

# random forest
mymodels_2 <- caret::train(myformula, data= mydata_train_treated, 
                               method="rf", silent = TRUE)
save(mymodels_2,file = "mymodels_rf.Rdata")

# J48 simple decision tree
mymodels_3 <- caret::train(myformula,
                       metric = "Kappa",
                       data = mydata_train_treated,
                       method = "J48",
                       trControl = fitControl)
.jcache(mymodels_3$finalModel$classifier)
save(mymodels_3,file = "mymodels_j48.Rdata")

# confusion matrices for test data
confusionMatrix(predict(mymodels_1,mydata_test_treated, type = "raw"),mydata_test_treated$class)
confusionMatrix(predict(mymodels_2,mydata_test_treated, type = "raw"),mydata_test_treated$class)
confusionMatrix(predict(mymodels_3,mydata_test_treated, type = "raw"),mydata_test_treated$class)

myensembledata_train_treated <- data.frame(
        mynnet = (predict(mymodels_1,mydata_train_treated, type = "prob")),
        myrf = (predict(mymodels_2,mydata_train_treated, type = "prob")),
        myJ48 = (predict(mymodels_3,mydata_train_treated, type = "prob")),
        class = mydata_train_treated$class)

mymodels_ensemble_treated <- caret::train(myformula, data= myensembledata_train_treated, 
                                        method = 'LogitBoost',
                                        metric = "Kappa",
                                        savePredictions = T)

save(mymodels_ensemble_treated,file = "myensemblemodel.Rdata")

# end parallel computing clusters
stopCluster(cl)

ml.end.time <- Sys.time()

ml.run.time <- round(difftime(ml.end.time, ml.start.time , units = 'secs'),1)

myensembledata_test_treated <- data.frame(
        mynnet = (predict(mymodels_1,mydata_test_treated, type = "prob")),
        myrf = (predict(mymodels_2,mydata_test_treated, type = "prob")),
        myJ48 = (predict(mymodels_3,mydata_test_treated, type = "prob"))
        )

# generate ensemble predictions
myensembledata_test_treated_ensemble_preds_treated <- (predict(mymodels_ensemble_treated,myensembledata_test_treated, type = "prob"))

# rename ensemble columns
myensembledata_test_treated_ensemble_preds_treated <- dplyr::rename(myensembledata_test_treated_ensemble_preds, ensemble.benign = benign, ensemble.malignant = malignant)

# add ensemble predictions to test data set predictions
myensembledata_test_treated <- cbind(myensembledata_test_treated,myensembledata_test_treated_ensemble_preds_treated)

# add the class
myensembledata_test_treated$class <- mydata_test_treated$class

# plot ROC curve of ensemble classifier
plot(roc(myensembledata_test_treated$class , myensembledata_test_treated$malignant, direction="<"))

mycfm_model_1 <- confusionMatrix(predict(mymodels_1, mydata_test_treated[,col(mydata_test_treated)-1]),
                                 mydata_test_treated$class,
                                 positive = "malignant")

mycfm_model_2 <- confusionMatrix(predict(mymodels_2, mydata_test_treated[,col(mydata_test_treated)-1]),
                                 mydata_test_treated$class,
                                 positive = "malignant")

mycfm_model_3 <- confusionMatrix(predict(mymodels_3, mydata_test_treated[,col(mydata_test_treated)-1]),
                                 mydata_test_treated$class,
                                 positive = "malignant")

mycfm_ensemble <- confusionMatrix(predict(mymodels_ensemble,
                                        myensembledata_test_treated[,col(myensembledata_test_treated)-1]),
                                        myensembledata_test_treated$class,
                                        positive = "malignant")

print(mycfm_ensemble)

ensemble_test_preds_class <- predict(mymodels_ensemble,
                                     myensembledata_test_treated[,col(myensembledata_test_treated)-1], type = "raw")

ensemble_test_preds_prob <- predict(mymodels_ensemble,
                                     myensembledata_test_treated[,col(myensembledata_test_treated)-1], type = "prob")

# ggpairs(subset(myensembledata_test, select = c(mynnet.malignant, myrf.malignant, myJ48.malignant, ensemble.malignant, class) ),ggplot2::aes(colour = class)) +
#         theme_bw()