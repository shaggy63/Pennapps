from IPython import get_ipython
ipython = get_ipython()
try:
    # Some issue irrelevant commands
    ipython.magic("matplotlib inline"); ipython.magic("matplotlib qt5") 
    ipython.magic("pylab --no-import-all")
except AttributeError:  # Not in an IPython environment
    pass
del ipython

# import libraries
import pandas as pd
import statsmodels.api as sm
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
from pandas_datareader import data as web
import bs4 as bs
import datetime as dt
import os
import pickle
import requests
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
#import ipdb

# save_sp500_tickers()
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.','-')
        tickers.append(ticker)
        tickers = [t.replace('\n', '') for t in tickers]
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2018, 7, 30)
    end = dt.datetime(2019, 7, 30)
    for ticker in tickers:
        #print(ticker)
        print()
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            #print('Already have {}'.format(ticker))
            print()
save_sp500_tickers()
#print (get_data_from_yahoo())
print()



df = DataReader('WMT', 'yahoo', '2018-07-30', '2019-07-30')['Close']
sp_500 = DataReader('^GSPC', 'yahoo', '2018-07-30', '2019-07-30')['Close']

# joining the closing prices of the two datasets 
monthly_prices = pd.concat([df, sp_500], axis=1)
monthly_prices.columns = ['WMT', '^GSPC']

# check the head of the dataframe
print(monthly_prices.head())

# calculate monthly returns
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row

# split dependent and independent variable
X = clean_monthly_returns['^GSPC']
y = clean_monthly_returns['WMT']
print (X)
print (y)

# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model 
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
print(results.summary())

# alternatively scipy linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

print(slope)

'''
plt.figure(figsize=(20,10))
X.plot()
y.plot()
plt.ylabel("Daily Returns")
plt.show()
'''

# Calculate the mean of x and y
Xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numerator and denominator of beta
df['xycov'] = (X.dropna() - Xmean)*(y.dropna() - ymean)
df['xvar'] = (X.dropna() - Xmean)**2


#Calculate beta and alpha
beta = df['xycov'].sum()/df['xvar'].sum()
alpha = ymean-(beta*Xmean)
print(f'alpha = {alpha}')
print(f'beta = {beta}')

# Generate Line
xlst = np.linspace(np.min(X),np.max(X),100)
ylst = np.array([beta*xvl+alpha for xvl in xlst])


# Plot
plt.scatter(X, y, alpha=0.5)
plt.scatter(X, y, color='r')
plt.scatter(y, X, color='b')
plt.plot(xlst,ylst,'k-')

plt.title('Percentage Returns for Stocks')
plt.xlabel('Company')
plt.ylabel('S&P 500')
plt.grid()
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
plt.show()

#ipdb.set_trace()

'''
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    for ticker in tickers:
        df = DataReader('%ticker', 'yahoo', '2018-07-30', '2019-07-30')['Close']
        sp_500 = DataReader('^GSPC', 'yahoo', '2018-07-30', '2019-07-30')['Close']
        
        # joining the closing prices of the two datasets 
        monthly_prices = pd.concat([df, sp_500], axis=1)
        monthly_prices.columns = ['%ticker', '^GSPC']
        
        # check the head of the dataframe
        print(monthly_prices.head())
        
        # calculate monthly returns
        monthly_returns = monthly_prices.pct_change(1)
        clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
        
        # split dependent and independent variable
        X = clean_monthly_returns['^GSPC']
        y = clean_monthly_returns['%ticker']
        print (X)
        print (y)
        
        # Add a constant to the independent value
        X1 = sm.add_constant(X)
        
        # make regression model 
        model = sm.OLS(y, X1)
        
        # fit model and print results
        results = model.fit()
        print(results.summary())
        
        # alternatively scipy linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        print(slope)
'''        
'''
        plt.figure(figsize=(20,10))
        X.plot()
        y.plot()
        plt.ylabel("Daily Returns")
        plt.show()
'''
'''
        
        # Calculate the mean of x and y
        Xmean = np.mean(X)
        ymean = np.mean(y)
        
        # Calculate the terms needed for the numerator and denominator of beta
        df['xycov'] = (X.dropna() - Xmean)*(y.dropna() - ymean)
        df['xvar'] = (X.dropna() - Xmean)**2
        
        
        #Calculate beta and alpha
        beta = df['xycov'].sum()/df['xvar'].sum()
        alpha = ymean-(beta*Xmean)
        print(f'alpha = {alpha}')
        print(f'beta = {beta}')
        
        # Generate Line
        xlst = np.linspace(np.min(X),np.max(X),100)
        ylst = np.array([beta*xvl+alpha for xvl in xlst])
        
        
        # Plot
        plt.scatter(X, y, alpha=0.5)
        plt.scatter(X, y, color='r')
        plt.scatter(y, X, color='b')
        
        plt.plot(xlst,ylst,'k-')
        
        plt.title('Percentage Returns for Stocks')
        plt.xlabel('Company')
        plt.ylabel('S&P 500')
        plt.grid()
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        plt.show()
        continue
'''
app = Flask(__name__)
arr = ['shreyway',1333]

@app.route('/')
def func():
    return jsonify(arr)

if __name__ == '__main__':
    app.run(debug=True, port = 8080)