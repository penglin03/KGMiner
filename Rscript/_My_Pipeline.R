# ---- Cleanup everything before start ----
rm(list = ls())
gc()

### Find true capital-state pairs from all possible capital-state pairs

# ---- GBSERVER API ----
source("./experimentAPI.R")

# ---- INPUT and CONFIGURATIONS ----

EDGE_TYPE_FILE = "../data/infobox.edgetypes" # Example : "../data/lobbyist.edgetypes"
INPUT_FILE = "../facts/state_capital.csv" # Example : "../facts/lobbyist/firm_payee.csv" col 1 and 2 are ids and 3 is label
CLUSTER_SIZE = 48 # Number of workers in gbserver
FALSE_PER_TRUE = 5
DISCARD_REL = 191
ASSOCIATE_REL = c(404)
max_depth = 3

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
dat.true <- read.csv(INPUT_FILE)

if (ncol(dat.true) < 3)
  dat.true$label <- T

# ---- Construct false labeled data -----
set.seed(233)

# TODO: reformat this so it is universal and file independent
dat.false <- rbind.fill(apply(dat.true, 1, function(x){
  candidates <- unique(dat.true[which(dat.true[,1] != x[1]), 2])
  candidates <- unlist(lapply(candidates, function(y){
    if(length(which(dat.true[,1] == x[1] & dat.true[,2] == y) != 0)) {
      return(NULL)
    }
    return(y)
  }))
  return(data.frame(src=x[1], 
                    dst=sample(candidates, FALSE_PER_TRUE),
                    label=F))
}))

colnames(dat.true) <- c("src","dst","label")
dat <- rbind(dat.true, dat.false)

elapsed.time <- data.frame()

## Test Method

experiment.fullpath.test <- eval.fullpath.test(dat, DISCARD_REL)
write.csv(experiment.fullpath.test$raw, "../result/city/capital_state_all.fullpath.test.csv", row.names=F)

print("FULL PATH")
print(experiment.fullpath.test$eval)

elapsed.time <- rbind(elapsed.time, data.frame(method="fullpath.test", 
                                               elapsed = experiment.fullpath.test$elapsed[3] * CLUSTER_SIZE / nrow(dat)))

experiment.test <- eval.test(dat, DISCARD_REL)
write.csv(experiment.test$raw, "../result/city/capital_state_all.test.csv", row.names=F)
print("PREDICATE PATH")
print(experiment.test$eval)

elapsed.time <- rbind(elapsed.time, data.frame(method="test", 
                                               elapsed = experiment.test$elapsed[3] * CLUSTER_SIZE / nrow(dat)))

q()
