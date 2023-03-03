###############################################################################
################ IC208: Programming for Finance  ##############################
################ Lecture 7: Portfolio theory in Python    #####################
###############################################################################
import os
import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt

path=r'____________'                                ##put_your_path_here
os.chdir(path)


tickers = ['AAL', 'AAPL', 'GM','JPM', 'MSFT', 'WMT']
prices = yf.download(tickers,'2015-1-1','2023-1-1')['Adj Close']
################# Save data in your local drive ###############################
#prices.to_csv('portfolio_example.csv')
#prices = pd.read_csv('portfolio_example.csv')
#prices['datetime'] = pd.to_datetime(prices['Date'])
#prices = prices.set_index('datetime')
#prices = prices.drop(columns=['Date'])
###############################################################################
prices.head(10)
#price.tail(10)
prices.plot()
##################### calculate daily returns  ################################
def log_return(dailyprice):
    return np.log(dailyprice / dailyprice.shift(1))
#### One way to loop through the dataframe prices
returns = pd.DataFrame()
for ticker in tickers:
    returns['{}'.format(ticker)] = log_return(prices['{}'.format(ticker)])

#### Another way
rets = log_return(prices)

# Portfolio valuation
## recall 
## tickers = ['AAL', 'AAPL', 'GM','JPM', 'MSFT', 'WMT']
## define weights vector for example we use [15%, 15%, 15%, 15%, 20%, 20%]
weights = np.array([0.15, 0.15, 0.15, 0.15, 0.20, 0.20])

port_return = np.sum(returns.mean()*weights)
port_variance = np.dot(weights.T, np.dot(returns.cov(), weights)) 

### annualise the portfolio's return and variance
port_return = 250*port_return
port_variance = port_variance*250
port_std = np.sqrt(port_variance)





################## Visualise Efficient frontier in portfolio theory ###########
n_assets = len(tickers)
weights = np.random.random(n_assets)
weights.sum()
weights /= np.sum(weights)
weights.sum()
weights

def port_return(weights):
    return np.sum(returns.mean()*weights)*252 

def port_std(weights):
    port_variance = np.dot(weights.T, np.dot(returns.cov()*252, weights))
    return np.sqrt(port_variance)

port_return(weights)
port_std(weights)



### We repeat this process in loop to have 100/1000/10000 …
### portfolios with random weights of the individual assets

portfolio_returns = []
portfolio_std = []

for i in range(1000):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    portfolio_returns.append(port_return(weights))
    portfolio_std.append(port_std(weights))

rets = np.array(portfolio_returns)
risks = np.array(portfolio_std)




## visualisation efficient frontier

Rf = 0.005
sharpe_ratio = (rets-Rf)/risks

plt.figure()
plt.scatter(risks, rets, c=sharpe_ratio, marker ='o')
plt.colorbar(label = 'Sharpe Ratio')
plt.show()
