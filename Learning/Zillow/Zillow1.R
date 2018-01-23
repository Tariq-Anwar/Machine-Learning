# Zillow Competition

require(tidyverse)

setwd("C:/Users/Nicol/Google Drive/Zillow")

#Data
#Properties_2016
prop <- read.csv("properties_2016.csv")
train <- read.csv("train_2016_v2.csv")
dict <- read.csv("dictionary.csv",header=FALSE) # 0=ID, 1= factor, 2=continous, 3= hybrid, 4=KEY ID

sample <- sample(1:nrow(prop), (.25)*nrow(prop), replace=FALSE)
data <- prop[sample,]
rm(prop)

head(dict)
# Categorical
cat.dex <- dict[,2] == 1
data[1:5,cat.dex]

# Categorical
cont.dex <- dict[,2] == 2
data[1:5,cont.dex]


## Exploration
names(data)
summary(data)
View(data[1:100,])


