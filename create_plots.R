# extract result files to results directory
unzip("results.zip", exdir="results")

.libPaths( c( .libPaths(), "Rlib") )

library(labeling, lib="Rlib")
library(crayon, lib="Rlib")
library(backports, lib="Rlib")
library(tzdb, lib="Rlib")
library(withr, lib="Rlib")
library(rstudioapi, lib="Rlib")
library(cli, lib="Rlib")
library(tidyverse, lib="Rlib")

# checks if directory for plots exists, if not creates it
if (!dir.exists("plots")){
  dir.create("plots")
} 

# path to directory with csv files
inputDir = "results"

# function to plot distribution of predictions
plot_distribution <- function(inputFile, inputDir){ 
  predictions <- read.csv(paste0(inputDir, "/", inputFile), header = TRUE, row.names=1)
  modelName = strsplit(inputFile,"_")[[1]][1]
  predictions$Class <- "positive"
  predictions[predictions$Prediction<0.5,'Class'] <- "negative"
  
  p <- ggplot(predictions, aes(x=Prediction, fill=Class)) +
    geom_density(alpha=.7) + scale_fill_manual("Label", values = c("#0073C2FF", "orange"))+ 
    theme_classic() + theme(axis.title = element_text(color = "black", size=16),
                            axis.text = element_text(color = "black", size=16),
                            legend.title = element_text(color = "black", size=16),
                            legend.text=element_text(color = "black", size=16)) +
    ggtitle(modelName) + labs(x = "Prediction probabilities", y = "Density")
  ggsave(paste0("plots/", modelName, "_distribution.png"), plot = p)
  
}


# function to plot histograms with frequency of scores within selected ranges
plot_ranges <- function(inputFile, inputDir){
  modelName = strsplit(inputFile,"_")[[1]][1]
  validation <- read.csv(paste0(inputDir, "/", inputFile), header=TRUE, row.names=1)
  
  validation$predictedClass <- "1"
  validation[validation$Prediction < 0.5,"predictedClass"] <- "0"
  
  validation$Result <- "Correct"
  validation[validation$Expectation != validation$predictedClass,"Result"] <- "Incorrect"
  
  # bins with score between 0 and 1, step = 0.1
  bin1A <- sum(validation[validation$Prediction>=0 & validation$Prediction <= 0.1,"Result"] =="Correct")
  bin1B <- sum(validation[validation$Prediction>=0 & validation$Prediction <= 0.1,"Result"] =="Incorrect")
  
  bin11A <- sum(validation[validation$Prediction>=0.1 & validation$Prediction <= 0.2,"Result"] =="Correct")
  bin11B <- sum(validation[validation$Prediction>0.1 & validation$Prediction <= 0.2,"Result"] =="Incorrect")
  
  bin2A <- sum(validation[validation$Prediction>0.2 & validation$Prediction <= 0.3,"Result"] =="Correct")
  bin2B <- sum(validation[validation$Prediction>0.2 & validation$Prediction <= 0.3,"Result"] =="Incorrect")
  
  bin22A <- sum(validation[validation$Prediction>0.3 & validation$Prediction <= 0.4,"Result"] =="Correct")
  bin22B <- sum(validation[validation$Prediction>0.3 & validation$Prediction <= 0.4,"Result"] =="Incorrect")
  
  
  bin3A <- sum(validation[validation$Prediction>0.4 & validation$Prediction <= 0.5,"Result"] =="Correct")
  bin3B <- sum(validation[validation$Prediction>0.4 & validation$Prediction <= 0.5,"Result"] =="Incorrect")
  
  bin33A <- sum(validation[validation$Prediction>0.5 & validation$Prediction <= 0.6,"Result"] =="Correct")
  bin33B <- sum(validation[validation$Prediction>0.5 & validation$Prediction <= 0.6,"Result"] =="Incorrect")
  
  
  bin4A <- sum(validation[validation$Prediction>0.6 & validation$Prediction <= 0.7,"Result"] =="Correct")
  bin4B <- sum(validation[validation$Prediction>0.6 & validation$Prediction <= 0.7,"Result"] =="Incorrect")
  
  
  bin44A <- sum(validation[validation$Prediction>0.7 & validation$Prediction <= 0.8,"Result"] =="Correct")
  bin44B <- sum(validation[validation$Prediction>0.7 & validation$Prediction <= 0.8,"Result"] =="Incorrect")
  
  
  bin5A <- sum(validation[validation$Prediction>0.8 & validation$Prediction <=0.9 ,"Result"] =="Correct")
  bin5B <- sum(validation[validation$Prediction>0.8 & validation$Prediction <= 0.9,"Result"] =="Incorrect")
  
  
  bin55A <- sum(validation[validation$Prediction>0.9 & validation$Prediction <=1 ,"Result"] =="Correct")
  bin55B <- sum(validation[validation$Prediction>0.9 & validation$Prediction <= 1,"Result"] =="Incorrect")
  
  
  
  df <- data.frame(value = rep(c("0-0.1","0.1-0.2", "0.2-0.3","0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1"), each=2 ),
                   result =rep(c("Correct", "Incorrect"), 5), count = c(bin1A/(bin1A+bin1B), bin1B/(bin1A+bin1B),bin11A/(bin11A+bin11B), bin11B/(bin11A+bin11B),
                                                                        bin2A/(bin2A+bin2B), 
                                                                        bin2B/(bin2A+bin2B), bin22A/(bin22A+bin22B),bin22B/(bin22A+bin22B),
                                                                        bin3A/(bin3A+bin3B), 
                                                                        bin3B/(bin3A+bin3B),bin33A/(bin33A+bin33B), bin33B/(bin33A+bin33B), 
                                                                        bin4A/(bin4A+bin4B), 
                                                                        bin4B/(bin4A+bin4B), bin44A/(bin44A+bin44B),bin44B/(bin44A+bin44B),
                                                                        bin5A/(bin5A+bin5B), 
                                                                        bin5B/(bin5A+bin5B), bin55A/(bin55A+bin55B),bin55B/(bin55A+bin55B)))
  # calculates percentage
  df$count <- round(df$count, digits=2)*100 
  
  df2 <- df %>%
    group_by(value) %>%
    arrange(value, desc(result)) %>%
    mutate(lab_ypos = cumsum(count) - 0.5 * count) 
  
  p <- ggplot(data = df2, aes(x = value, y = count)) +
    geom_col(aes(fill = result), width = 0.7)+
    geom_text(aes(y = lab_ypos, label = count, group =result), color = "white", size=3) +
    scale_fill_manual("Label", values = c("#0073C2FF", "orange")) + 
    theme_classic() + theme(axis.title = element_text(color = "black", size=12),
                            axis.text = element_text(color = "black", size=12), 
                            legend.title = element_text(color = "black", size=12),
                            legend.text=element_text(color = "black", size=12)) +
    ggtitle(paste0("Number of correct/incorrect predictions from ", modelName, " model")) + labs(x = "Prediction probability", y = "Number of predictions [%]")
  
  ggsave(paste0("plots/", modelName, "_ranges.png"), height=6.25, width=10.5, plot = p)
  
}


# create plot with final accuracies from Kaggle (achieved by all models)
final_plot <- function(){
  nb <- 73.28
  voting <- 85.84
  logistic <- 79.84
  bert <- 88.20
  svm <- 81.96
  lstm <- 86.12
  cnn <- 84.48
  rf <- 84.86
  average <- 86.40
  
  df <- data.frame(value = c(73.28, 85.84, 79.84, 88.20, 81.96, 86.12, 84.48, 84.86, 86.40),
                   names = c("Naive Bayes", "Voting", "Log Reg", "Bert", "SVM", "LSTM", "CNN", 
                             "Random Forest", "Average"))
 

  df_ord <- df[order(df$value), ]  # sort
  df_ord$names <- factor(df_ord$names, levels = df_ord$names)  # to retain the order in plot
  
  p <- ggplot(df_ord, aes(x=names, y=value)) + 
    geom_bar(stat="identity", width=.5, fill="tomato3") + 
    theme_classic() + geom_text(aes(y = 50, label = value), color = "white", size=4) +
    theme(axis.text.x = element_text(size=12, angle=65, color="black", vjust=0.5), axis.text.y = element_text(size=12, color="black"),
          axis.title = element_text(size=12, color="black")) +
    ggtitle("Final accuracies") + labs(x = "Model", y = "Accuracy [%]")
  
  
  ggsave("plots/final_results.png", height=6.25, width=10.5, plot = p)
  
  
  
}


# create plots for all models
files_exact <- list.files(path=inputDir, pattern="*_result_exact.csv")
files_validation <- list.files(path=inputDir, pattern="*validation_set.csv")

for (f in files_exact){
  plot_distribution(f, inputDir)
}

for (f in files_validation){
  plot_ranges(f, inputDir)
}

final_plot()



