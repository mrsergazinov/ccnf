set.seed(1234)

# generate sequences
series = t(sapply(1:10000, function(x) {arima.sim(n=500, 
                                                list(ar = c(0.9)), 
                                                sd=sqrt(0.0001),
                                                n.start = 400)}))

# transform sequences
trans_series = series
a = runif(length(trans_series), 0, 1)
b = runif(length(trans_series), 0, 1) 
c = runif(length(trans_series), 0, 100)
trans_series = exp(a*trans_series+b)+c*trans_series

# save 
df = data.frame(series)
write.csv(df, file='series.csv')
df = data.frame(trans_series)
write.csv(df, file='trans_series.csv')