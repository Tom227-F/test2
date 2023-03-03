### Seminar Week 8 ###
print('test')
import numpy as np
import pandas as pd
import yfinance as yf

tickers = ['^GSPC']

data=pd.DataFrame()


data['^GSPC']= yf.download(tickers, '2015-01-01', '2022-12-30')['Adj Close']

def log_return(dailyprice):
    return np.log(dailyprice/dailyprice.shift(1))
data['ret_^GSPC'] = log_return(data)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(211)
plt.plot(data['ret_^GSPC'], label='return')
plt.legend()

plt.show()

import numpy as np
import pandas as pd
import yfinance as yf

tickers = ['^GSPC']

data=pd.DataFrame()


data['^GSPC']= yf.download(tickers, '2015-01-01', '2022-12-30')['Adj Close']

def log_return(dailyprice):
    return np.log(dailyprice/dailyprice.shift(1))
data['ret_^GSPC'] = log_return(data)


mus, sigmas = [],[]

for y in range (2015,2023):
    mask = (data.index.year==y)
    
    mus.append(data['ret_^GSPC'][mask].mean()*252)
    sigmas.append(data['ret_^GSPC'][mask].std()*np.sqrt(252))
    
    print ('year:',y,',mean:', mus[-1],',std:',sigmas[-1])

import numpy as np
import pandas as pd
import yfinance as yf

def log_return(dailyprice):
    return np.log(dailyprice/dailyprice.shift(1))

tickers = ['AAPL','MSFT','TSLA','GOOGL','AMZN','ADBE','ORCL','IBM','META','TSM']
data = yf.download(tickers, '2020-01-01','2022-12-30')['Adj Close']

returns=pd.DataFrame()

for t in tickers:
    returns[t]=log_return(data[t])
    
weights=np.ones(10)
print(weights)

weights /= np.sum(weights)
print(weights)
print(tickers)
print(returns[tickers])
port_return = np.sum(returns[tickers].mean()*weights)*252
print(port_return)

port_variance = np.dot(weights.T,np.dot(returns[tickers].cov(),weights))
print('Portfolio variance',port_variance*252)
print('Portfolio std', port_variance*np.sqrt(252))

weights = np.ones(10)
weights /= np.sum(weights)
port_return = np.sum(returns[tickers].mean()*weights)*252
port_variance=np.dot(np.dot(weights.T,returns[tickers].cov()),weights)*252

print ('Portfolio Return:', port_return,'\nportfolo variance:',port_variance)



















