
## ARIMA 
- stands for Autoregressive Integrated Moving Average models
- Alternative name: Box-Jenkins
- A forecasting technique that projects the future values of a series based on its own inertia
- main application = short term forecasting requiring at least 40 historical data points
- Work best when: 
	+ Data exhibits a stable or consistent pattern
	+ Minimum amount of outliers

## Stationary
- Check stationary first
- The data should also show a constant variance in its fluctuations over time

## Differencing
- an excellent way of transforming a nonstationary series to a stationary one
- done by subtracting the observation in the current period from the previous one

## Autocorrelations
- Indicates how a data series is related to itself over time

## Models
- ARIMA attempts to describe the movement in a stationary time series as a function of "autoregressive and moving avg"
	+ AR(autoregressive)
	+ MA(moving avg)
- Autoregressive Models: 
	$$X(t) = A(1)*X(t-1) + A(2)*X(t-2) +... + A(p)*X(t-p) + E(t)$$
	+ X(t): the time-series
	+ X(t-p):  time series lagged p
	+ A(p): autoregressive parameters
	+ E(t): the error term of the model
- Moving Average Models:
	$$X(t) = -B(1) * E(t-1) + E(t)$$
	+ B(1): MA of order 1
- Mixed model
	$$ARIMA(p,d,q)$$
	+ p: number of lag observations included in the model, also called the lag order
	+ d: number of times that the raw observations are differenced, also called the degree of differencing
	+ q: size of the moving average window, also called the order of moving average.
## Reference
[https://www.quora.com/What-is-ARIMA](https://www.quora.com/What-is-ARIMA)


