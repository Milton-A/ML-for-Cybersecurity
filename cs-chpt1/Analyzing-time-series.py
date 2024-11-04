#Begin by generating a time series

from random import random
time_series = [2 * x + random() for x in range(1, 100)]

#Plot your data:


import matplotlib.pyplot as plt
plt.plot(time_series)
#plt.show()

#There is a large variety of techniques we can use to predict the consequent value of a time series:
    #Autoregression (AR):

from statsmodels.tsa.ar_model import AutoReg
model = AutoReg(time_series, lags=3)
model_fit = model.fit()
y = model_fit.predict(len(time_series), len(time_series))

print()
print(y)

    #Moving average (MA):
    
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(time_series, order=(0, 1, 1))
model_fit = model.fit()
y = model_fit.predict(len(time_series), len(time_series))
print()
print(y)

    #Simple exponential smoothing (SES):

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model = SimpleExpSmoothing(time_series)
model_fit = model.fit()
y = model_fit.predict(len(time_series), len(time_series))
print()
print(y)