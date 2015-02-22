# Random Forest on Weight Lifting Exercises Dataset
Marvin Lav  
Saturday, February 21, 2015  


##Introduction

Weight Lifting Exercises can often be done wrong. The dataset compiles a lot of users doing the exercises correctly and incorrectly, giving them different identifiers of letters A through E. These gym people wore devices that could record a lot of acceleration data. This data could be used to identify if the exercise were exercise A, B, C, D, or E. Given that we had a lot of data, the approach used was Random Forest. Random forest is a very nice black box, that will predict very well given many variables and data.

##Method

The first part is opening up the file to see what variables can be used. In this aspect, I used Excel to look through the file. There were many NA's for the columns. These only had a number if the new_window = "yes". There was no clear documentation on what this meant, and this data was filtered out for the training.

Some of the columns were timestamps, names, and tracking number. These would not have an effect on classifying A,B,C,D, or E and were not factored in. All in all, about 50 variables could be used to generate the predictor.


```r
#Loading libraries
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# Load data and Remove the "new_window" line
pml_training <- read.csv("pml-training.csv")
clean_pml <- pml_training[pml_training$new_window == "no",]
```

###Creating the cross validation set

Have 40% of the dataset be the training. Then use this model on the other 60% of the dataset to see how well it worked. Also, random forest takes a while to run. If the training set is reduced, then it can be completed much faster. In addition, having a smaller set can reduce overfitting. clean_pml.train will be used to train, and clean_pml.test will be used for testing


```r
set.seed(20150222)
clean_pml$randu = runif(19216,0,1)
clean_pml.train = clean_pml[clean_pml$randu < .4,]
clean_pml.test = clean_pml[clean_pml$randu >= .4,]
```

###Choosing the variables and creating the model

As stated before, ~50 variables are used. As long as it is meaningful, a variable can be factored into the model. A row mean of the sigificance was taken of the model, to see which variable played the most significance.


```r
#Create the Random Forest Model using the testing set
pml.model.rf = randomForest(classe ~ roll_belt   + pitch_belt+  yaw_belt+  total_accel_belt+  gyros_belt_x+	gyros_belt_y+	gyros_belt_z+	accel_belt_x+	accel_belt_y+	accel_belt_z+	magnet_belt_x+	magnet_belt_y+	magnet_belt_z+	roll_arm+	pitch_arm+	yaw_arm+	total_accel_arm+	gyros_arm_x+	gyros_arm_y+	gyros_arm_z+	accel_arm_x+	accel_arm_y+	accel_arm_z+	magnet_arm_x+	magnet_arm_y+	magnet_arm_z+	roll_dumbbell+	pitch_dumbbell+	yaw_dumbbell+	total_accel_dumbbell+	gyros_dumbbell_x+	gyros_dumbbell_y+	gyros_dumbbell_z+	accel_dumbbell_x+	accel_dumbbell_y+	accel_dumbbell_z+	magnet_dumbbell_x+	magnet_dumbbell_y+	magnet_dumbbell_z+	roll_forearm+	pitch_forearm+	yaw_forearm+	gyros_forearm_x+	gyros_forearm_y+	gyros_forearm_z+	accel_forearm_x+	accel_forearm_y+	accel_forearm_z+	magnet_forearm_x+	magnet_forearm_y+	magnet_forearm_z, data = clean_pml.train,importance=TRUE)

pml.model.rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ roll_belt + pitch_belt + yaw_belt +      total_accel_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z +      accel_belt_x + accel_belt_y + accel_belt_z + magnet_belt_x +      magnet_belt_y + magnet_belt_z + roll_arm + pitch_arm + yaw_arm +      total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z +      accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x +      magnet_arm_y + magnet_arm_z + roll_dumbbell + pitch_dumbbell +      yaw_dumbbell + total_accel_dumbbell + gyros_dumbbell_x +      gyros_dumbbell_y + gyros_dumbbell_z + accel_dumbbell_x +      accel_dumbbell_y + accel_dumbbell_z + magnet_dumbbell_x +      magnet_dumbbell_y + magnet_dumbbell_z + roll_forearm + pitch_forearm +      yaw_forearm + gyros_forearm_x + gyros_forearm_y + gyros_forearm_z +      accel_forearm_x + accel_forearm_y + accel_forearm_z + magnet_forearm_x +      magnet_forearm_y + magnet_forearm_z, data = clean_pml.train,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 1.15%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2151    4    0    1    1    0.002782
## B   16 1416    9    1    0    0.018031
## C    0   11 1294    7    0    0.013720
## D    0    0   24 1236    2    0.020602
## E    0    0    4    7 1387    0.007868
```

As can be seen, the out of bounds error is ~1%. So I expect in the cross validation that the error rate is about 1%


```r
#Creating the significance dataframe
imps <- varImp(pml.model.rf)
significance <- data.frame(rowMeans(imps))
names(significance) <- "significance"
df <- data.frame(rownames(significance),significance)

#Shows the top 6 factors with most significance
head(df[order(significance,decreasing=TRUE),])
```

```
##                   rownames.significance. significance
## roll_belt                      roll_belt        38.83
## yaw_belt                        yaw_belt        37.26
## magnet_dumbbell_z      magnet_dumbbell_z        36.78
## pitch_belt                    pitch_belt        35.43
## magnet_dumbbell_y      magnet_dumbbell_y        33.20
## pitch_forearm              pitch_forearm        31.31
```

The above is to show that there are some variables that have more weight than others. If processing power is of concern, or some variables need to be reduced, a second random forest with less predictors can be performed. As it is, the random forest is completed and can be cross-validated with the Test set.

###Cross validation with test set


```r
#Running the test set through the predictor, and getting the confusion matrix
clean_pml.test$pred.pml.rf = predict(pml.model.rf, clean_pml.test, type="response")
print(table(clean_pml.test$classe, clean_pml.test$pred.pml.rf))
```

```
##    
##        A    B    C    D    E
##   A 3307    4    2    0    1
##   B   40 2223   13    0    0
##   C    0   21 2015    4    0
##   D    0    0   44 1839    2
##   E    0    0    1   12 2117
```

```r
print(prop.table(table(clean_pml.test$classe, clean_pml.test$pred.pml.rf),1))
```

```
##    
##             A         B         C         D         E
##   A 0.9978877 0.0012070 0.0006035 0.0000000 0.0003018
##   B 0.0175747 0.9767135 0.0057118 0.0000000 0.0000000
##   C 0.0000000 0.0102941 0.9877451 0.0019608 0.0000000
##   D 0.0000000 0.0000000 0.0233422 0.9755968 0.0010610
##   E 0.0000000 0.0000000 0.0004695 0.0056338 0.9938967
```

There are two tables, one with the numbers of obserbance and the other in percentage. As seen in the diagonal matrix of the proportional table, a majority of it is .99 or .98, so it is off by 2-3 percentage at most. Averaged over the 5 classe, the error is about 1%.

The same prediction model was used on the 20 unknowns. They were successfully interpreted.
