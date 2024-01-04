library(tidyverse)
library(dplyr)
library(magrittr)
library(ggplot2)
library(ggthemes)
library(ppcor)
library(reshape2)
library(groupdata2)
library(knitr)       # kable()

#library(gglasso)
library(glmnet)
library(ggsci)
library(viridis)
library(ggExtra)
library(kableExtra)
library(xtable)
library(ggrepel)
library(scales)
library(car)
library(pROC)
library(patchwork)      # Multi-plot alignment
library(brainconn)
library(igraph)
library(network)
library(brainconn)
library(brainGraph)
library(RColorBrewer)
library(networkD3)
library(ggpubr)
library(DescTools)
library(plotly)
#library(data.table)
library(rsample)   
library(purrr)
library(dplyr)
library(ggplot2)
library(scales)
library(mlbench)
library(kernlab)
library(sessioninfo)
options(warn = -1)
rm(list=ls())


#####################################
### DATA DIR
#####################################
input_dir <- "../data/input_data"
dest_dir <- "../data/result_data"


#####################################
### LOAD DATA
#####################################

X <- read_csv(paste(input_dir, "X.csv", sep = "/"), col_types = cols(.default = col_double()))
Y <- read_csv(paste(input_dir, "Y.csv", sep = "/"), col_types = cols(.default = col_double())) %>% rename(y = x)
Y %>% group_by(y) %>% count()

# rest data subjects
filter <- data.frame(HCPID = Sys.glob(paths = paste(input_dir, "connectivity_matrix/REST1/*", sep = "/"))) %>%
  mutate(HCPID = str_extract(HCPID, "(?<=sub-)\\d+") %>% paste0("_fnca"))


dvs <- read_csv(paste(input_dir, "LL.csv", sep = "/"), #read_csv("../data/actr_maxLL.csv", 
                col_types = cols(
                  .default = col_double(),
                  HCPID = col_character(),
                  best.model = col_character(), 
                  diff.LL = col_double()
                )) %>% 
  # find common subject ID that both rest and task data are available
  inner_join(filter, by = "HCPID") %>%
  # exclude bad subjects that are hard to distinguish by two models
  #anti_join(bad_subj, by = "HCPID") %>%
  mutate(BestModel = best.model, LL.diff = diff.LL) %>%
  dplyr::select(HCPID, BestModel, LL.diff) %>%
  arrange(HCPID) %>%
  mutate(Y = as.numeric(as.factor(BestModel)) -1,  # model1 = 0, model2 = 1
         HCPID = factor(HCPID), 
         BestModel=factor(BestModel))



# based on proportion of two labels
W <- Y$y
W[W == 0] <- mean(Y)
W[W == 1] <- (1-mean(Y))

# normalize maxLL diff
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
W <- min_max_norm(dvs$LL.diff)

merged_data <- cbind(merge(X, Y, by = "...1")[, -1], W)


#####################################
### SAMPLING FUNCTION
#####################################

sampling <- function(sample_method, merged_data) {
  
  if (sample_method == "shuffle") {
    sampled_data <- merged_data
    shuffle_rows <- sample(1:length(rownames(merged_data)))
    sampled_data$y <- merged_data$y[shuffle_rows]
    sampled_data$W <- merged_data$W[shuffle_rows]
  } else if (sample_method == "up") {
    sampled_data <- groupdata2::upsample(
      data = merged_data, 
      id_method = "n_rows_c",
      cat_col = "y")
  } else if (sample_method == "up-shuffle") {
    sampled_data <- groupdata2::upsample(
      data = merged_data, 
      id_method = "n_rows_c",
      cat_col = "y")
    shuffle_rows <- sample(1:length(rownames(sampled_data)))
    sampled_data$y <- sampled_data$y[shuffle_rows]
    sampled_data$W <- sampled_data$W[shuffle_rows]
  } else if (sample_method == "down") {
    sampled_data <- groupdata2::downsample(
      merged_data,
      id_method = "n_rows_c",
      cat_col = "y")
    
  } else if (sample_method == "none") {
    sampled_data <- merged_data
    
  } else {
    print("Invalid sample_method. Please choose from: up, down, up-shuffle, shuffle, none")
  }
  
  return(sampled_data)
  
}

save_result_data <- function(data, file_name, n = 0) {
  data$n <- n
  write.table(data, file = file_name, sep = ",", 
              col.names = !file.exists(file_name), 
              row.names = FALSE, append = TRUE)
}


#####################################
### PLOT FUNCTION
#####################################

plot.lambda <- function(fit.cv) {
  png(filename = "./figure/lambda.png", bg = "transparent", 
      width = 800, height = 800, 
      units = "px", res = 150)
  
  print(plot(fit.cv))
  dev.off()  
}

plot.beta_histogram <- function(best_model) {
  betas <- as.matrix(best_model$beta) 
  res <- ggdensity(data = data.frame(betas) %>% filter(s0 != 0), 
                   x = "s0", 
                   fill = "forestgreen") +
    labs(x = "Coefficients", y = "Values") +
    ggtitle(paste("Histogram of non-zero coefficients", "[ N =", sum(betas != 0), "]"))
  
  png(filename = "./figure/beta_histogram.png", bg = "transparent", 
      width = 800, height = 800, 
      units = "px", res = 150)
  
  print(res) 
  dev.off() 
}

plot.roc <- function(prediction_table) {
  
  rocobj <- roc(prediction_table$Y, prediction_table$predictions) 
  
  png(filename = "./figure/roc.png",  bg = "transparent", 
      width = 1000 , height = 800, 
      units = "px", res = 150)
  
  ggroc(rocobj, col = "red") +
    geom_point(aes(y = rocobj$sensitivities, x = rocobj$specificities), col = "red", size = 4, alpha = 0.5) +
    ggtitle(paste("AUC ROC Curve:", rocobj$auc[1])) +
    xlab("Specificity (FPR)") + ylab("Sensitivity (TPR)") + 
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "grey", linetype = "dashed") +
    theme_minimal()
  
  dev.off()  
}

plot.accuracy <- function(prediction_table) {
  wcomparison <- prediction_table %>% 
    rename(Observed = Y,
           Predicted = Yp,
           DiscretePredicted = predictions) %>%
    mutate(Accuracy = ifelse(DiscretePredicted == Observed,
                             "Correct", 
                             "Misclassified")) %>% drop_na()
  
  accuracy_score <- round(100 * sum(wcomparison$Accuracy == "Correct") / nrow(wcomparison), 2)

  png(filename = "./figure/accuracy.png",  bg = "transparent", 
      width = 1000, height = 800, 
      units = "px", res = 150) 
  
  res <- ggplot(wcomparison, aes(x=Predicted, y=Observed, 
                                  col=Accuracy)) +
    geom_point(size=4, alpha=0.6, 
               position= position_jitter(height = 0.02)) +
    geom_abline(intercept = 0, slope = 1, 
                col="red",
                linetype="dashed") +
    scale_color_d3() +
    theme_pander() +
    
    theme(legend.position = "right") +
    guides(col=guide_legend("Classification")) +
    coord_fixed(xlim=c(0, 1), ylim=c(0, 1)) +
    annotate("text", x=0.3, y=0.7,
             label=paste("Accuracy (",
                         length(Y),
                         ") = ",
                         accuracy_score,
                         "%",
                         sep="")) +
    ylab("Observed Strategy") +
    xlab("Predicted Strategy") +
    ggtitle(paste("Predicted vs. Observation")) +
    theme(legend.position = "bottom")
  
  
  ggMarginal(res, fill = "grey", alpha = 0.75,
             type = "density", col = "darkgrey", margins = "both")
  print(res) 
  dev.off()   
}


#####################################
### LASSO PIPELINE
#####################################

pipeline <- function(merged_data = merged_data, sample_method = "up", save_data = FALSE, n = FALSE) {
  
  # Deal with imbalanced data
  sampled_data <- sampling(sample_method = sample_method, merged_data = merged_data)
  
  
  Y.sampled <- sampled_data$y
  W.sampled <- sampled_data$W
  X.sampled <- sampled_data[, !names(sampled_data) %in% c("y", "W")]
  
  
  # Sample 80% of the indices for training
  indices <- 1:nrow(X.sampled)
  train_indices <- sample(indices, 0.8 * length(indices), replace = FALSE)
  
  # Create training and testing sets
  X_train <- X.sampled[train_indices, ]
  Y_train <- Y.sampled[train_indices]
  W_train <- W.sampled[train_indices]
  
  X_test <- X.sampled[-train_indices, ]
  Y_test <- Y.sampled[-train_indices]
  W_test <- W.sampled[-train_indices]
  
  
  # LOO CV to find optimal lambda
  fit.cv <- cv.glmnet(y = Y_train,
                      x = as.matrix(X_train),
                      alpha=1,
                      family = "binomial",
                      weights = W_train,
                      type.measure = "class",
                      standardize=T,
                      nfolds=length(Y_train),
                      grouped = F, 
                      keep = T)
  
  best_model <- glmnet(y = Y_train,
                       x = as.matrix(X_train),
                       alpha=1,
                       lambda = fit.cv$lambda.min,
                       weights = W_train,
                       family = "binomial",
                       type.measure = "class",
                       standardize = T)
  
  # evaluation
  predictions <- predict(best_model, 
                         newx = as.matrix(X_test),
                         weights =W_test,
                         s=fit.cv$lambda.min,
                         type="class", family = "binomial")%>% as.numeric()
  Yp <- predict(best_model, 
                newx = as.matrix(X_test),
                weights =W_test,
                s=fit.cv$lambda.min,
                type="response", family = "binomial")%>% as.numeric()
  
  
  # Compute accuracy
  accuracy_score <- mean(predictions == Y_test)
  roc_score <- auc(roc(Y_test, predictions))
  
  
  # Save data
  beta_table <- data.frame(as.matrix(best_model$beta), row.names = colnames(X)[2:length(X)])
    
  prediction_table <- data.frame(Yp = Yp, 
                                 predictions = predictions, 
                                 Y = Y_test, 
                                 row.names = 1:length(Y_test))
  score_table <- data.frame(lambda = fit.cv$lambda.min, 
                             accuracy_score = accuracy_score,
                             roc_score = roc_score)
  
  if (save_data) {
    print("... SAVING DATA ...")
    
    save_result_data(data = beta_table, file_name = paste(dest_dir, sample_method, "beta_data.csv", sep = "/"), n = n)
    save_result_data(data = prediction_table, file_name = paste(dest_dir, sample_method, "prediction_table.csv", sep = "/"), n = n)
    save_result_data(data = score_table, file_name = paste(dest_dir, sample_method, "score_table.csv", sep = "/"), n = n)
  }
  
  if (n == FALSE) {
    print("... SAVING PLOTS ...")
    
    plot.beta_histogram(best_model = best_model)
    plot.lambda(fit.cv = fit.cv)
    plot.accuracy(prediction_table = prediction_table)
    plot.roc(prediction_table = prediction_table)
    
  }
  
  return(list(beta_table = beta_table, score_table = score_table, prediction_table = prediction_table))

}


#####################################
### RUN MODEL
#####################################
set.seed(0)
result = pipeline(merged_data = merged_data, sample_method = "up", save_data = TRUE, n = FALSE)

# Access the returned variables from the list
print(result$score_table)


#####################################
### RUN MODEL N TIMES
#####################################

# Set the loop to run n=100 times
n <- 100

# Create an empty data frame to store the results
score_table <- data.frame(accuracy_score = numeric(), roc_score = numeric())
score_table$n <- 1:n

for (i in 1:n) {
  result = pipeline( merged_data = merged_data, sample_method = "up", save_data = TRUE, n = i)
  print(paste("...FINISHED...", i))
  print(result$score_table )
}


#####################################
### PLOT
#####################################

read.csv(paste(dest_dir, "up-shuffle", "score_table.csv", sep = "/")) %>% melt(
  id.vars = "n", 
  measure.vars = c("accuracy_score", "roc_score"), 
  variable.name = "score", 
  value.name = "value") %>%
  ggboxplot(x = "score", y="value", fill = "score", title = "Testing Score with Up-Shuffle (Repeated N = 100 times)", 
            xlab = "score type", ylab = "mean score (N = 100)") + ylim(0,1)

