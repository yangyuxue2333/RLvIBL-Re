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
### PLOT BETA
#####################################
load(file = "../data/input_data/I.RData")
load(file = "../data/input_data/order.RData")

B <- read_csv(paste(dest_dir, "up", "beta_data.csv", sep = "/"))
conn_betas <- as_tibble(data.frame(index=rep(x=I$index, times=max(B$n)), Beta=B$s0, n=B$n))


join_connectom <- function(conn_betas) {
  connectome <- data.frame()
  for (i in 1:max(conn_betas$n)) {
    
    conn_betas_i <- conn_betas %>% filter(n==i)
    
    connectome_i <- order %>%
      filter(index %in% I$index) %>%
      inner_join(conn_betas_i) %>%
      dplyr::select(-censor2) %>%
      filter(Beta != 0) %>% 
      separate(connection, c("connection1", "connection2"))%>%
      separate(network, sep = "-", c("network1", "network2"), remove = F) %>% 
      mutate(network1 = ifelse(str_detect(network, pattern = "-1-"), -1, network1)) %>%
      mutate(connection_type = ifelse(network1==network2, "Within", "Between")) %>%
      arrange(index)
    
    connectome_i$n <- i
    
    
    # HARD CODE
    connectome_i[connectome_i$network=="-1-5","network2"] <- "5"
    connectome_i[connectome_i$network=="-1-7","network2"] <- "7"
    connectome_i[connectome_i$network=="-1--1","network2"] <- "-1"
    connectome_i[connectome_i$network=="-1-11","network2"] <- "11"
    connectome_i[connectome_i$network=="-1-12","network2"] <- "12"
    connectome_i[connectome_i$network=="-1-13","network2"] <- "13"
    
    connectome <- rbind(connectome, connectome_i)
  }
  
  return(connectome)
}
connectome <- join_connectom(conn_betas)

connectome %>% 
  mutate(beta_sign = ifelse(Beta >0, "+", "-")) %>%
  ggdotchart(x = "network_names", y = "Beta",
             color = "beta_sign",                                # Color by groups
             palette = c("steelblue", "tomato"), # Custom color palette
             rotate = TRUE,
             facet.by = "connection_type", 
             sort.by.groups = F,
             sort.val = "desc",          # Sort the value in descending order
             sorting = "descending",                       # Sort value in descending order
             add = "segments",                             # Add segments from y = 0 to dots
             add.params = list(color = "lightgray", size = 2), # Change segment color and size
             group = "connection_type",                                # Order by groups
             dot.size = 3,                                 # Large dot size
             title = paste("Lasso Connection Weights:", dim(connectome)[[1]]),
             ggtheme = theme_pander()) +
  geom_hline(yintercept = 0, linetype = 2, color = "black")


## How many connections are survived? N = 70
conn_betas %>%
  group_by(n) %>%
  summarise(num_nonzeros = sum(Beta != 0)) %>%
  ggdensity(x = "num_nonzeros", fill = "salmon", color = "gray")

conn_betas %>%
  filter(Beta !=0 ) %>%
  mutate(sign = ifelse(Beta>0, "+", "-")) %>%
  ggdensity(x="index", fill = "sign")


connectome %>% 
  mutate(sign = ifelse(Beta>0, "+", "-")) %>%
  group_by(sign) %>%
  ggdensity(x = "Beta", fill = "sign")
  
  




