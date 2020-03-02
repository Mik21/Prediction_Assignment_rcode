
#Course : Practical Machine Learning
#Assignment : Final Project 

#Submitted by Michael 3/1/2020 

#1. Description
# This report as a complemet of this Practical machin Learning course.
#The overview of this assignment to perform a prediction. The main goal 
#of the project is to predict the manner in whivh 6 participants performed 
#some exercise as described below. The "classe" varaible in the training set. 
#The machine learning algorithm described here is applied to test 
#cases available in the test data and predictions are submitted in the
#format discribed  in the course project prediction Quiz for Automated grading.

#2. Background( in order to have a clear understanding,
   # we copy it from the  discribtion given in the assignment)

#Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now 
#possible to collect a large amount of data about personal activity relatively
#inexpensively. These type of devices are part of the quantified self movement 
#a group of enthusiasts who take measurements about themselves regularly to 
#improve their health, to find patterns in their behavior, or because they are 
#tech geeks. One thing that people regularly do is quantify how much of 
#a particular activity they do, but they rarely quantify how well they do it. 
#In this project, your goal will be to use data from accelerometers on the belt,
#forearm, arm, and dumbell of 6 participants. They were asked to perform barbell
# lifts correctly and incorrectly in 5 different ways. More information is 
#available from the website here: http://groupware.les.inf.puc-rio.br/har
 #(see the section on the Weight Lifting Exercise Dataset). 

#3. Reading Data and Descriptive Analysis ( Plot/ table)
 # i) Data 
#Data 
#The training data for this project are available here: 
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#The test data are available here:
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
#The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

#Source : 
#Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative 
#Activity Recognition of Weight Lifting Exercises. 
#Proceedings of 4th International Conference in Cooperation with SIGCHI
#(Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

#My deep gratitude goes to these Authors who allow us to use their data 
#and to do some research.

#A discription of the dataset from the Authors websight is 
#"Six young health participants were asked to perform one set of 10 repetitions
#of the Unilateral Dumbbell Biceps Curl in five different fashions:
#exactly according to the specification (Class A), throwing 
#the elbows to the front (Class B), lifting the dumbbell only halfway (Class C),
#lowering the dumbbell only halfway (Class D) and throwing the hips 
#to the front (Class E).
#Class A corresponds to the specified execution of the exercise,
 #while the other 4 classes correspond to common mistakes. 
#Participants were supervised by an experienced weight lifter to make
 #sure the execution complied to the manner they were supposed to simulate.
 #The exercises were performed by six male participants aged between 
#20-28 years, with little weight lifting experience. We made sure that
 #all participants could easily simulate the mistakes in a safe and 
#controlled manner by using a relatively light dumbbell (1.25kg).
# Read more on this websight below:
 #http://groupware.les.inf.puc-rio.br/har#ixzz6FTLOqr8p"

#3. Analysis 
 #  First I download the dataset from the link above to my desktop then
#I read it from the excel file. Once we read succesfully we split the traing set
#into another subset as a training data set and test set in order to build my model.
#Before we build oour model we need to perform descriptive analysis and data cleaning
#in order to determine which varaibles will be included in the final model.

# Packages 
install.packages("knitr")
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(mlbench)
set.seed(23339)

# loading the training set and test set 

training <-read.csv("C:/Users/mikha/Documents/Coursera_Practice_machin_learning/week4/pml_training.csv", header =T)
testing <-read.csv("C:/Users/mikha/Documents/Coursera_Practice_machin_learning/week4/pml_testing.csv", header =T)

# Spliting the training set in to  subtraining and subtesting

inTrain<-createDataPartition(y=training$classe, p = 0.7, list = FALSE)
subtraining<-training[inTrain,]
subtesting<-training[-inTrain,]

dim(subtraining)
# [1] 13737   160
dim(subtesting)
#[1] 5885  160

# checking for NA
head(which(is.na(subtraining)))
# Both datasets have 160 variables. Those variables have plenty of NA, 
# that can be removed with the cleaning procedures

# remove variables with Nearly Zero Variance
NZV <- nearZeroVar(subtraining)
subtraining<- subtraining[, -NZV]
subtesting <- subtesting[, -NZV]
dim(subtraining)
# [1] 13737   103
dim(subtesting)
# 5885  103

# removing those varaibles with missing Value
M<-sapply(subtraining, function(x) sum(is.na(x)))>0
subtraining<- subtraining[, M == FALSE]
subtesting <- subtesting[, M== FALSE]

dim(subtraining)

# [1] 13737    59
dim(subtesting)
#[1] 5885   59
# Removing variables not used in our model

subtraining<- subtraining[, -(1:5)]
subtesting <- subtesting[, -(1:5)]

dim(subtraining)
# [1] 13737    54
dim(subtesting)
#[1] 5885   54



# calculate correlation matrix
correlationMatrix <- cor(subtraining[,-54])
# summarize the correlation matrix
print(correlationMatrix)


# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.8)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#A correlation matrix is created from these attributes and highly correlated
#attributes are identified. Here are some of the attributes with high correlation
# [1] 11  2 10 37  9 12 35 22 26 46 32 34 19

# 4  Prediction  Model Building 
 i)  Decision Tree
# Modele building using rpar method to fit a model
set.seed(23339)
modFit<- train(classe~., data=subtraining, method="rpart")
print(modFit$finalModel)

#n= 13737 

#node), split, n, loss, yval, (yprob)
 #     * denotes terminal node

 #1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
 #  2) roll_belt< 130.5 12563 8667 A (0.31 0.21 0.19 0.18 0.11)  
 #    4) pitch_forearm< -33.95 1116    7 A (0.99 0.0063 0 0 0) *
 #    5) pitch_forearm>=-33.95 11447 8660 A (0.24 0.23 0.21 0.2 0.12)  
 #     10) roll_forearm< 124.5 7304 4807 A (0.34 0.25 0.15 0.19 0.068)  
 #       20) magnet_dumbbell_y< 439.5 5998 3561 A (0.41 0.19 0.18 0.17 0.059) *
 #       21) magnet_dumbbell_y>=439.5 1306  632 B (0.046 0.52 0.032 0.29 0.11) *
 #     11) roll_forearm>=124.5 4143 2866 C (0.07 0.21 0.31 0.21 0.21) *
 #  3) roll_belt>=130.5 1174   10 E (0.0085 0 0 0 0.99) *

# prediction on Test dataset
predDT<-predict(modFit, newdata=subtesting, type="class")
confMatDT<-confusionMatrix(predDT, subtesting$classe)
confMatDT

# RESULTS FROM DECISION TREE
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1434  130   10   51   59
         B   70  623   10  104  172
         C    9   73  866  149   95
         D  154  253  135  627  149
         E    7   60    5   33  607

Overall Statistics
                                         
               Accuracy : 0.7064         
                 95% CI : (0.6946, 0.718)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.6293         
                                         
 Mcnemar's Test P-Value : < 2.2e-16      

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.8566   0.5470   0.8441   0.6504   0.5610
Specificity            0.9406   0.9250   0.9329   0.8596   0.9781
Pos Pred Value         0.8515   0.6364   0.7265   0.4757   0.8525
Neg Pred Value         0.9429   0.8948   0.9659   0.9262   0.9082
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2437   0.1059   0.1472   0.1065   0.1031
Detection Prevalence   0.2862   0.1664   0.2025   0.2240   0.1210
Balanced Accuracy      0.8986   0.7360   0.8885   0.7550   0.7696



ii) Boosting using gml method in the caret package
# model fit
set.seed(23339)
control<-trainControl(method = "repeatedcv", number = 3, repeats = 1)
modFit<-train(classe~., data=subtraining, method = "gbm", trControl = control, verbose = FALSE)

   # RESULTSS FROM BOOSTING
#A gradient boosted model with multinomial loss function.
# 150 iterations were performed.
#There were 53 predictors of which 53 had non-zero influence.

# prediction on Test dataset
pred<-predict(modFit, newdata=subtesting)
confMat<-confusionMatrix(pred, subtesting$classe)
confMat
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1674    7    0    2    1
         B    0 1121    7    6    1
         C    0    6 1017   13    4
         D    0    5    2  941    6
         E    0    0    0    2 1070

Overall Statistics
                                          
               Accuracy : 0.9895          
                 95% CI : (0.9865, 0.9919)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9867          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9842   0.9912   0.9761   0.9889
Specificity            0.9976   0.9971   0.9953   0.9974   0.9996
Pos Pred Value         0.9941   0.9877   0.9779   0.9864   0.9981
Neg Pred Value         1.0000   0.9962   0.9981   0.9953   0.9975
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2845   0.1905   0.1728   0.1599   0.1818
Detection Prevalence   0.2862   0.1929   0.1767   0.1621   0.1822
Balanced Accuracy      0.9988   0.9906   0.9932   0.9867   0.9942




iii) Random forest 
# Model building using the rf method in the caret package to fit a model
set.seed(23339)
controlRF<- trainControl(method= "cv", number=3, verboseIter=FALSE)
modFitRF<-train(classe~., data= subtraining, method="rf",trControl=controlRF)

modFitRF$finalModel

Call:
 randomForest(x = x, y = y, mtry = param$mtry) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 27

        OOB estimate of  error rate: 0.22%
Confusion matrix:
     A    B    C    D    E  class.error
A 3904    1    0    0    1 0.0005120328
B    9 2646    3    0    0 0.0045146727
C    0    2 2394    0    0 0.0008347245
D    0    0    8 2243    1 0.0039964476
E    0    0    0    5 2520 0.0019801980


# prediction on Test dataset
predictRF<-predict(modFitRF newdata= subtesting)
confMatRF<-confusionMatrix(predictRandForest, subtesting$classe)
confMatRF

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1674    0    0    0    0
         B    0 1135    4    0    0
         C    0    3 1022    4    0
         D    0    1    0  960    2
         E    0    0    0    0 1080

Overall Statistics
                                         
               Accuracy : 0.9976         
                 95% CI : (0.996, 0.9987)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.997          
                                         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9965   0.9961   0.9959   0.9982
Specificity            1.0000   0.9992   0.9986   0.9994   1.0000
Pos Pred Value         1.0000   0.9965   0.9932   0.9969   1.0000
Neg Pred Value         1.0000   0.9992   0.9992   0.9992   0.9996
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2845   0.1929   0.1737   0.1631   0.1835
Detection Prevalence   0.2845   0.1935   0.1749   0.1636   0.1835
Balanced Accuracy      1.0000   0.9978   0.9973   0.9976   0.9991

#5 Final prediction on the Test set that kept for validation (This dataset is given)
 From above we obtained the accuracy for 
i ) Decision tree =  0.7064  
ii) Boosting      =  0.9895
iii) random Forest=  0.9976 

#Based on the accuracy , the random forest is the best model fit .So we select the 
#Random forest to pridict 20 test cases available in the test data.


#a) Random Forest
predTEST_RF<-predict(modFitRF, newdata=testing)
predTEST_RT 
#[1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

#b) Boosting 

predTEST_Bt<-predict(modFit, newdata=testing)
predTEST_Bt
#[1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

#c) Decision Tree 

predTEST_Dt<-predict(modFit, newdata=testing)
predTEST_Dt

#1] C A C A A C C A A A C C C A C A A A A C
#Levels: A B C D E

#From fitting all the models to the test data,except that of dicision tree,
#we got the same result from Random forest and boosting.Their accuracy is 
#prety much the same. 




