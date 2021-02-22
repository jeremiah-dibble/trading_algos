# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:34:51 2021

@author: Jerem
"""

import pandas as pd
import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt

#Now lets take our unrealized and realizede gains from Apex clearing
unreal_df = pd.read_csv('unrealized.csv')
real_df = pd.read_csv('realized.csv')
#Combine them 
total_df = unreal_df.append(real_df)
#Now sort by date
total_df['Trade Date'] = pd.to_datetime(total_df['Trade Date'])
total_df = total_df.sort_values(by='Trade Date')
#total_df.index = total_df['Trade Date']
unwanted_columns = ['CUSIP', 'Type', 'Description', \
                   'Short Term Gain/Loss','Long Term Gain/Loss',\
                       'Disallowed Loss', 'LRM', 'Long/Short Position',\
                           '1256 Contract', 'Market Discount', 'Wash Sale']
total_df = total_df.drop(unwanted_columns, axis=1)
#Optional change index to the trade date
#date= total_df['Trade Date']
#total_df = total_df.drop(["Trade Date"], axis = 1)
#total_df.index = date
#day1 = str(total_df.index[0])[:10]
#pd.DataFrame.reset_index(total_df)
#pd.DataFrame.reset_index(date)
#total_df['Trade Date'] = date['date']

#Now lets grab the returns of our stocks from yahoo finance
tickers = total_df.Symbol.unique()
tickers_list = tickers.tolist()
tickers_list[tickers_list.index('BRKB')] = 'BRK-B'
tickers = np.array(tickers_list)

day1 = str(total_df['Trade Date'][0])[4:14]

# Fetch the data
bench_ticker = 'SPY'
data = yf.download(tickers_list, day1)['Adj Close']
data = data.fillna(0)
pdata = yf.download(tickers_list, day1)['Adj Close'].pct_change()
pdata = data.fillna(0)
bench = yf.download(bench_ticker, day1)['Adj Close']
abench = np.array(bench)

tickers = data.columns
weights = pd.DataFrame((np.zeros(np.shape(data))), index=data.index, columns=data.columns)
snp_spend = pd.DataFrame(np.zeros((np.shape(data)[0],1)), index=data.index, columns=['SNP'])
basedate = pd.Timestamp('2019-12-04')


total_df['time since'] = (total_df['Trade Date'] - basedate).dt.days
total_df = pd.DataFrame.reset_index(total_df)
short_df = total_df.drop(['Buy/Sell', 'index'], axis=1)
    
atotal = np.array(short_df)
#i=0
#trading_days = data.index.tolist()
#for date in total_df['Trade Date']:
 #   print(trading_days.index(date))
#    atotal.T[-1][i] = trading_days.index(date)
#    i += 1
    
for trade in atotal:
    [stock, time, price, quantity, cost, timesince]=trade
    if (stock == 'BRKB'):
        stock = 'BRK-B'

    weights[stock][time] += quantity
    snp_spend['SNP'][time] += (cost*np.sign(quantity)/bench[time])
aspend = np.array(snp_spend) 
aweight = np.array(weights)
i = 0
for i in range(1, np.shape(aweight)[0]):
    aweight[i] += aweight[i-1]
    aspend[i] += aspend[i-1]
    
weights = pd.DataFrame(aweight, index=data.index, columns=data.columns)
snpw = pd.DataFrame(aspend, index=data.index, columns=['SNP'])
aspend = aspend.reshape((-1,))
snp_port = aspend*abench

plt.plot(data.index, snp_port, 'b')

adata = np.array(data)
full_returns = adata@aweight.T
returns = np.diag(adata@aweight.T)
plt.plot(data.index, returns, 'r')
plt.legend(['Benchmark', 'Personal'])
plt.xticks(rotation=45)

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Time matched portfolio comparision")