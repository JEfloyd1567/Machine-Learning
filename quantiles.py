list_continua = []

for var in list_continua:
  q1 = data[var].quantile(0.25)
  q3 = data[var].quantile(0.75)
  IQR = q3 - q1
  outliers_q1 = data.index[data[var] < q1 - 2*IQR]
  data.drop(labels=outliers_q1, inplace=True)
  outliers_q3 = data.index[data[var] > q3 + 2*IQR]
  data.drop(labels=outliers_q3, inplace=True)