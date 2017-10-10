# ---- Cleanup everything before start ----
rm(list = ls())
gc()

### Find true capital-state pairs from all possible capital-state pairs

# ---- GBSERVER API ----
source("./_My_ExperimentAPI.R")

# ---- INPUT and CONFIGURATIONS ----

EDGE_TYPE_FILE = "../data/infobox.edgetypes" # Example : "../data/lobbyist.edgetypes"
## INPUT_FILE = "../facts/state_capital2.csv" # Example : "../facts/lobbyist/firm_payee.csv" col 1 and 2 are ids and 3 is label
CLUSTER_SIZE = 48 # Number of workers in gbserver
max_depth = 3

DISCARD_REL = 25

# ---- Load edge type file ----

mapfile <- read.csv(EDGE_TYPE_FILE, sep="\t", header=F)
mapfile$V1 <- as.numeric(mapfile$V1)
mapfile$V2 <- as.character(mapfile$V2)

# ---- Init workers ----

cl <- makeCluster(CLUSTER_SIZE) 
clusterExport(cl = cl, varlist=c("adamic_adar", "semantic_proximity", "ppagerank", "heter_path",  "max_depth",
                                 "preferential_attachment", "katz", "pcrw", "heter_full_path", "meta_path",
                                 "multidimensional_adamic_adar", "heterogeneous_adamic_adar",
                                 "connectedby", "rel_path", "truelabeled", "falselabeled", "str_split",
                                 "as.numeric", "request","DISCARD_REL"), envir = environment())

# ---- Load input data ----
## dat <- read.csv(INPUT_FILE)

relation.list <- read.csv("../data/gfc_input_relations.tsv", head=FALSE, sep="\t")
colnames(relation.list) <- c("relation", "discard_rel")

relation.list

cat(paste("Relation", "Accuracy", "Precision", "Recall", "F1\n", sep = "\t"))

for (i in 1:nrow(relation.list)) {
  rel.str <- relation.list[i, 1]
  rel.int <- relation.list[i, 2]
  
  # print(rel.int)
  DISCARD_REL = rel.int
  clusterExport(cl = cl, varlist=c("DISCARD_REL"), envir = environment())

  train <- paste("../data/groundtruth/", rel.str, "_train.tsv", sep="")
  test <- paste("../data/groundtruth/", rel.str, "_test.tsv", sep="")

  datatrain <- read.csv(train, head = FALSE, sep="\t")
  datatest <- read.csv(test, head = FALSE, sep="\t")

  colnames(datatrain) <- c("src","dst","label")
  colnames(datatest) <- c("src","dst","label")

  # Either full paths or predicate paths.
  featuremerged <- extract.predicatepaths(rbind(datatrain, datatest), rel.int)
  
  if (ncol(featuremerged) == 1) {
    cat(paste(rel.str, round(0, digits=5), round(0, digits=5), round(0, digits=5), round(0, digits=5), "\n", sep = "\t"))
  } else {
  
    ## featuremerged <- extract.fullpaths(rbind(datatrain, datatest), rel.int)
      
    featurestrain <- featuremerged[1:nrow(datatrain), ]
    featurestest <- featuremerged[(nrow(datatrain) + 1):nrow(featuremerged), ]
    
    # print(dim(featurestrain))
    # print(dim(featurestest))
  
    modeltrain <- Logistic(as.factor(label) ~ .,featurestrain)
    # evaltest <- evaluate_Weka_classifier(modeltrain, newdata=featurestest, cost=NULL, numFolds = 0, complexity = T, class = T, seed = NULL)
    # 
    # TP <- evaltest$confusionMatrix["TRUE", "TRUE"]
    # FP <- evaltest$confusionMatrix["FALSE", "TRUE"]
    # TN <- evaltest$confusionMatrix["FALSE", "FALSE"]
    # FN <- evaltest$confusionMatrix["TRUE", "FALSE"]
    # accuracy <- (TP + TN) / (TP + FP + FN + TN)
    # precision <- TP / (TP + FP)
    # if (is.nan(precision)) {
    #   precision <- 0
    # }
    # recall <- TP / (TP + FN)
    # if (is.nan(recall)) {
    #   recall <- 0
    # }
    # f1 <- 2 * TP / (2 * TP + FP + FN)
    # if (is.nan(f1)) {
    #   f1 <- 0
    # }
    
    predict_test <- predict(modeltrain, newdata = featurestest)
    
    actual_true <- which(featurestest$label == TRUE)
    actual_false <- which(featurestest$label == FALSE)
    predicted_true <- which(predict_test == TRUE)
    predicted_false <- which(predict_test == FALSE)
    
    TP <- length(intersect(actual_true, predicted_true))
    FP <- length(intersect(actual_false, predicted_true))
    FN <- length(intersect(actual_true, predicted_false))
    TN <- length(intersect(actual_false, predicted_false))
    
    acc <- (TP + TN) / (TP + FP + FN + TN)
    prec <- TP / (TP + FP)
    recl <- TP / (TP + FN)
    f1 <- 2 * (prec * recl) / (prec + recl)
    
    acc <- ifelse(is.nan(acc), 0, acc)
    prec <- ifelse(is.nan(prec), 0, prec)
    recl <- ifelse(is.nan(recl), 0, recl)
    f1 <- ifelse(is.nan(f1), 0, f1)
    
    # print(paste(acc, prec, recl, f1))
    cat(paste(rel.str, round(acc, digits=5), round(prec, digits=5), round(recl, digits=5), round(f1, digits=5), "\n", sep = "\t"))
    
    output_df <- data.frame(featurestest$label, predict_test, row.names = NULL)
    colnames(output_df) <- c("actual", "predict")
    
    write.table(output_df, file = paste("../result/", rel.str, "_predicts.tsv", sep = ""), quote = FALSE, sep = "\t", row.names = FALSE)
    write.csv(featurestrain, paste("../result/", rel.str, "_train.csv", sep=""), row.names=FALSE)
    write.csv(featurestest, paste("../result/", rel.str, "_test.csv", sep=""), row.names=FALSE)
  }
}

# ---- Construct false labeled data -----

# elapsed.time <- data.frame()

## Test Method

# experiment.fullpath.test <- eval.fullpath.test(dat, DISCARD_REL)
# write.csv(experiment.fullpath.test$raw, "../result/city/capital_state_all.fullpath.test.csv", row.names=F)
# 
# print("FULL PATH")
# print(experiment.fullpath.test$eval)
# 
# elapsed.time <- rbind(elapsed.time, data.frame(method="fullpath.test", 
#                                                elapsed = experiment.fullpath.test$elapsed[3] * CLUSTER_SIZE / nrow(dat)))

## experiment.test <- eval.test(dat, DISCARD_REL)
## write.csv(experiment.test$raw, "../result/city/capital_state_all.test.csv", row.names=F)
## print("PREDICATE PATH")
## print(experiment.test$eval)

## elapsed.time <- rbind(elapsed.time, data.frame(method="test", 
##                                                elapsed = experiment.test$elapsed[3] * CLUSTER_SIZE / nrow(dat)))

q()
