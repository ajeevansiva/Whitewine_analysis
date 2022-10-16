install.packages("Metrics")
install.packages("MLmetrics")
install.packages("neuralnet")
install.packages("tsDyn")
install.packages("tidypredict")

library("neuralnet")
library("Metrics")
library("MLmetrics")
library("tsDyn")

# Reading the Csv file #

UoW_load <- read.csv("UoW_load.csv", header = TRUE)
View(UoW_load)

# Extracting data for testing and training
training_data = head(UoW_load, n =430)
testing_data = tail(UoW_load, n =70)

# Normalizing Data
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))}

# scaling using the normalization function

normalized_data <- as.data.frame(lapply(UoW_load[,2:4], normalize))
normalized_data <- cbind(normalized_data, UoW_load[c(4)])


# Changing column names
names(normalized_data)[1] <- "variable_1"
names(normalized_data)[2] <- "variable_2"
names(normalized_data)[3] <- "norm_pred"
names(normalized_data)[4] <- "pred"

# taking scaled values for testing and training
norm_testdata = tail(normalized_data, n =70)
norm_traindata = head(normalized_data, n =430)

# un normalize function
unnormalizing <- function(x, min, max) { 
  return( (max - min)*x + min )
}
View(norm_traindata)


# Taking the maximum and minimum value
min <- min(normalized_data[4])
max <- max(normalized_data[4])

######## AR Approach ###########################

## Model 1 ##

NNModel_1<- neuralnet(norm_pred~variable_1+variable_2 ,hidden=c(3,4) , data = norm_traindata 
                      ,linear.output=TRUE)
plot(NNModel_1)

#Evaluation model performance
modelResult1 <- predict(NNModel_1, norm_testdata[1:2])
modelResult1

renorm_pred_val1 <- unnormalizing(modelResult1, min, max)
renorm_pred_val1 = unlist(as.list(renorm_pred_val1),recursive=F)
renorm_pred_val1

## Model 2 ##

NNModel_2<- neuralnet(norm_pred~variable_1+variable_2 ,hidden=c(10,30,10) , data = norm_traindata 
                      ,linear.output=TRUE)
plot(NNModel_2)

#Evaluation model performance
modelResult2 <- predict(NNModel_2, norm_testdata[1:2])
modelResult2

renorm_pred_val2 <- unnormalizing(modelResult2, min, max)
renorm_pred_val2 = unlist(as.list(renorm_pred_val2),recursive=F)
renorm_pred_val2


## Model 3 ##

NNModel_3<- neuralnet(norm_pred~variable_1+variable_2 ,hidden=c(10,50,25,10) , data = norm_traindata 
                      ,linear.output=TRUE)
plot(NNModel_3)

#Evaluation model performance
modelResult3 <- predict(NNModel_3, norm_testdata[1:2])
modelResult3

renorm_pred_val3 <- unnormalizing(modelResult3, min, max)
renorm_pred_val3 = unlist(as.list(renorm_pred_val3),recursive=F)
renorm_pred_val3

########### NARX Approach  ###########

## Model 1 ##

narx.model1<- nnetTs(norm_traindata[c(3)],m=5, size=3,steps=30)
narx.model1
renorm_pred_val4 <- unnormalizing(predict(narx.model1,steps=5,n.ahead=20), min, max)
renorm_pred_val4 = unlist(as.list(renorm_pred_val4),recursive=F)
renorm_pred_val4

plot.ts(renorm_pred_val4)
plot.ts(normalized_data[c(2)])

## Model 2 ##

narx.model2<- nnetTs(norm_traindata[c(3)], m = 4, size=3,steps=20)
narx.model2
renorm_pred_val5 <- unnormalizing(predict(narx.model2,steps=5,n.ahead=20), min,  max)

renorm_pred_val5 = unlist(as.list(renorm_pred_val5),recursive=F)
renorm_pred_val5
plot.ts(renorm_pred_val5)

## Model 3 ##

narx.model3<- nnetTs(norm_traindata[c(3)], m = 5, size=8,steps=10)
narx.model3
renorm_pred_val6 <- unnormalizing(predict(narx.model3,steps=5,n.ahead=20), min, max)

renorm_pred_val6 = unlist(as.list(renorm_pred_val6),recursive=F)
renorm_pred_val6
plot.ts(renorm_pred_val6)


###### RMSE , MSE and MAPE  ############

## Model 1 ##

#RMSE
RMSE(renorm_pred_val1,testing_data[,4])
#MSE
MSE(renorm_pred_val1,testing_data[,4])
#MAPE
mape(renorm_pred_val1,testing_data[,4])

## Model 2 ##

#RMSE
RMSE(renorm_pred_val2,testing_data[,4])
#MSE
MSE(renorm_pred_val2,testing_data[,4])
#MAPE
mape(renorm_pred_val2,testing_data[,4])

## Model 3 ##

#RMSE
RMSE(renorm_pred_val3,testing_data[,4])
#MSE
MSE(renorm_pred_val3,testing_data[,4])
#MAPE
mape(renorm_pred_val3,testing_data[,4])

### Plotting ###

## Model 1 ##
# regression line #

plot(x=testing_data[,4], y = renorm_pred_val1, col = "red", 
     main = 'Real vs Predicted')
abline(0, 1, lwd = 2)
plot.ts(x = testing_data[,4] ,y = renorm_pred_val1)
abline(0, 1, lwd = 2,col = "red")

## Model 2 ##
# regression line #

plot(x=testing_data[,4], y = renorm_pred_val2, col = "red", 
     main = 'Real vs Predicted')
abline(0, 1, lwd = 2)
plot.ts(x = testing_data[,4] ,y = renorm_pred_val2)
abline(0, 1, lwd = 2,col = "red")

## Model 3 ##
# regression line #

plot(x=testing_data[,4], y = renorm_pred_val3, col = "red", 
     main = 'Real vs Predicted')
abline(0, 1, lwd = 2)
plot.ts(x = testing_data[,4] ,y = renorm_pred_val3)
abline(0, 1, lwd = 2,col = "red")

# Plotting Prediction Graph #

plot(testing_data[,4] , ylab = "Predicted vs Expected", type="l", col="green" )
par(new=TRUE)
plot(renorm_pred_val1, ylab = " ", yaxt="n", type="l", col="red" ,main='Predicted 
Value Vs Desired Value Graph - Model 1')
legend("topright",
       c("Expected","Predicted"),
       fill=c("green","red")
)

plot(testing_data[,4] , ylab = "Predicted vs Expected", type="l", col="green" )
par(new=TRUE)
plot(renorm_pred_val2, ylab = " ", yaxt="n", type="l", col="red" ,main='Predicted 
Value Vs Desired Value Graph - Model 2')
legend("topright",
       c("Expected","Predicted"),
       fill=c("green","red")
)

plot(testing_data[,4] , ylab = "Predicted vs Expected", type="l", col="green" )
par(new=TRUE)
plot(renorm_pred_val3, ylab = " ", yaxt="n", type="l", col="red" ,main='Predicted 
Value Vs Desired Value Graph - Model 3')
legend("topright",
       c("Expected","Predicted"),
       fill=c("green","red")
)
