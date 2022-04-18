# load libs and data
library(tidyverse)
library(lubridate)
library(latentcor)
load("SleepList.Rda")

# select only non-na rows and concat them into a matrix
out = matrix(ncol = ncol(SleepList[[i]]$dexcom_sleep))
for (i in 1:length(SleepList)){
  # select rows with more than 80% non-missing data
  idx = which(rowSums(!is.na(SleepList[[i]]$dexcom_sleep)) == ncol(SleepList[[i]]$dexcom_sleep))
  out = rbind(out, SleepList[[i]]$dexcom_sleep[idx, , drop = FALSE])
}
out = out[-1, ]
row.names(out) = NULL; colnames(out) = NULL
corr = latentcor(X = out,
          types = c("con"),
          method = "original",
          use.nearPD = TRUE)
write.csv(out, file = "cgmSleep_noNA.csv")
write.csv(corr$R, file = "cgmSleep_timecorr.csv")