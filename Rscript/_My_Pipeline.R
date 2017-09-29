# ---- Cleanup everything before start ----
rm(list = ls())
gc()

### Find true capital-state pairs from all possible capital-state pairs

# ---- GBSERVER API ----
source("./_My_ExperimentAPI.R")

# ---- INPUT and CONFIGURATIONS ----

EDGE_TYPE_FILE = "../data/infobox.edgetypes" # Example : "../data/lobbyist.edgetypes"
## INPUT_FILE = "../facts/state_capital2.csv" # Example : "../facts/lobbyist/firm_payee.csv" col 1 and 2 are ids and 3 is label
CLUSTER_SIZE = 4 # Number of workers in gbserver
max_depth = 3

DISCARD_REL = 191

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

for (i in 1:nrow(relation.list)) {
  rel.str <- relation.list[i, 1]
  rel.int <- relation.list[i, 2]
  train <- paste("../data/groundtruth/", rel.str, "_train.tsv", sep="")
  test <- paste("../data/groundtruth/", rel.str, "_test.tsv", sep="")

  datatrain <- read.csv(train, sep="\t")
  datatest <- read.csv(test, sep="\t")

  colnames(datatrain) <- c("src","dst","label")
  colnames(datatest) <- c("src","dst","label")

  # or use full paths: extract.fullpaths(datatrain, rel.int)
  featuremerged <- extract.predicatepaths(rbind(datatrain, datatest))

  featurestrain <- featuremerged[1:nrow(datatrain), ]
  featurestest <- featuremerged[(nrow(datatrain) + 1):nrow(featuremerged), ]

  write.csv(featurestrain, "train.csv")
  write.csv(featurestest, "test.csv")

  modeltrain <- Logistic(label~.,featurestrain)
  evaltest <- evaluate_Weka_classifier(modeltrain, newdata=featurestest, cost=NULL, numFolds = 0, complexity = T, class = T, seed = NULL)

  print(evaltest)
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
