library(caret)
library(tidyverse)
library(leaps)
library(ggplot2)
library(lattice)
library(reshape2)
library(MASS)
library(ggcorrplot)
library(corrplot)
library(plotmo)
library(keras)
library(kableExtra)
library(modelr)
library(psych)
library(Rmisc)
library(plyr)
library(dplyr)
library(gridExtra)
library(scales)
library(rpart)
library(yardstick)
library(cluster)
library(NbClust)
library(factoextra)
library(dplyr)

library(readxl)
Whitewine_v2 <- read_excel("C:/Users/Ajeevan Sivanandhan/Desktop/DM & ML/ML_CW/Whitewine_v2.xlsx")
View(Whitewine_v2)

boxplot(Whitewine_v2) ### boxplot before removing outliers
summary(Whitewine_v2)

##### Removing Outliers ####
outliers = c()
for ( i in 1:11 ) {
  stats = boxplot.stats(Whitewine_v2[[i]])$stats
  bottom_outlier_rows = which(Whitewine_v2[[i]] < stats[1])
  top_outlier_rows = which(Whitewine_v2[[i]] > stats[5])
  outliers = c(outliers , top_outlier_rows[ !top_outlier_rows %in% outliers ] )
  outliers = c(outliers , bottom_outlier_rows[ !bottom_outlier_rows %in% outliers ] )
}

newdata = Whitewine_v2[-outliers, ] ## New data set after removing outliers.

boxplot(newdata) ### boxplot after removing outliers

#removing the 12th column of the data set because it is the output class. 
finalDataset <- newdata[,-12]
#normalizing the data(Scaling data).
wineDataset_final <- scale(finalDataset)
#displaying the final scaled and cleaned dataset.
view(wineDataset_final)
#boxploting the final data set.
boxplot(wineDataset_final)
#checking the dimensions of the dataset.
dim(wineDataset_final)


## Defining Cluster centers

########################### Elbow Method ########

#finding the number of clusters(k) automatically by the elbow method.
fviz_nbclust(wineDataset_final, kmeans, method = "wss")+
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow Method")


########################### Silhoutte Method ########

fviz_nbclust(wineDataset_final, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

########### Finding Number of Clusters Manually#####

# Initialize total within sum of squares error: wss
wss <- 0

for (i in 1:10) {
  km.out <- kmeans(wineDataset_final, centers = i)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}
wss
# Plot total within sum of squares vs. number of clusters

plot(1:10, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

######## K-Means Clustering ##########

# When k=2

km2 <- kmeans(wineDataset_final, centers =2, nstart =25)
km2

km2.clusters <- km2$cluster

fviz_cluster(km2, geom = "point", data = wineDataset_final)+ ggtitle("k = 2")

#BSS value
km2$betweenss
#TSS Value
km2$totss
#WSS Value
km2$withinss
#Value of BSS/TSS 
km2$betweenss/km2$totss


# When k=3

km3 <- kmeans(wineDataset_final, centers =3, nstart =25)
km3

km3.clusters <- km3$cluster

fviz_cluster(km3, geom = "point", data = wineDataset_final)+ ggtitle("k = 3")

#BSS value
km3$betweenss
#TSS Value
km3$totss
#WSS Value
km3$withinss
#Value of BSS/TSS 
km3$betweenss/km3$totss


# When k=4

km4 <- kmeans(wineDataset_final, centers =4, nstart =25)
km4

km4.clusters <- km4$cluster

fviz_cluster(km4, geom = "point", data = wineDataset_final)+ ggtitle("k = 4")

#BSS value
km4$betweenss
#TSS Value
km4$totss
#WSS Value
km4$withinss
#Value of BSS/TSS 
km4$betweenss/km4$totss

#getting the 12 column of the data set

qualityColumn <-factor(newdata$quality)
wineQuality <-as.numeric(qualityColumn)

cm2 <- confusionMatrix(as.factor(km2$cluster), as.factor(wineQuality))
cm2

cm3 <- confusionMatrix(as.factor(km3$cluster), as.factor(wineQuality))
cm3

cm4 <- confusionMatrix(as.factor(km4$cluster), as.factor(wineQuality))
cm4

################## Applying PCA ############################

#getting the cleaned data set to check the principal components.
pca_Data <- wineDataset_final
View(pca_Data)

# Proceeding with principal components

PCA <- prcomp(finalDataset, center = T, scale. = T)
plot(PCA)
plot(PCA, type='l')
summary(PCA)

# creating a new dataset using the cumulative proportion greater than 96%

newDataset <- as.data.frame(PCA$x[,1:9])
summary(newDataset)

## Apply K-Means to new PCA Data ###

# When k=2(Winning Clusters method) ###

kmnew <- kmeans(newDataset, centers =2, nstart =25)
kmnew

kmnew.clusters <- kmnew$cluster

fviz_cluster(kmnew, geom = "point", data = newDataset)+ ggtitle("k = 2")

#BSS value
kmnew$betweenss
#TSS Value
kmnew$totss
#WSS Value
kmnew$withinss
#Value of BSS/TSS 
kmnew$betweenss/kmnew$totss
#centers
kmnew$centers

