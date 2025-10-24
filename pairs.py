import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import itertools
import requests
from collections import deque

##
## TODO list:
## 1. Produce presentable panel of summary statistics and visualizations
## 2. Parallelize
## 3. Improve console GUI
##


url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {'User-Agent':'Mozilla/5.0'}
window_size = 100
t = 200
ENTRY_Z = 2.5
EXIT_Z = 0.25
SLIPPAGE_RATE = 0.0002
TRANSACTION_COST = 0.0005
hedge_dict = {}
historical = {}
prev_position = {}
returns_list = []


def pull_data(start_date, end_date):
    response = requests.get(url, headers=headers)
    table = pd.read_html(response.text)
    tickers = table[0]['Symbol'].to_list()
    tickers = [t.replace('.', '-') for t in tickers]
    data = yf.download(tickers, start=start_date, end=end_date)
    
    prices = data['Close'].dropna(axis=1).drop(columns=['GOOG'])
    returns = np.log(prices / prices.shift(1)).dropna()
    corr = returns.corr()
    return prices, corr


def find_pairs(prices, corr):
    pairs = [(a,b) for a,b in itertools.combinations(prices.columns, 2) if abs(corr.loc[a,b])>0.85]
    pairs = [(a,b) for a,b in pairs if coint(prices[a],prices[b])[1] < 0.05]
    
    for stockA,stockB in pairs:
        hedge_ratio = np.cov(prices[stockA],prices[stockB])[0,1] / np.var(prices[stockB])
        hedge_dict[(stockA,stockB)] = hedge_ratio
    return pairs

def compute_spread(stockA, stockA_price, stockB, stockB_price):
    spread = stockA_price - hedge_dict[(stockA,stockB)] * stockB_price
    return spread

def check_spread_deviation(mu, sd, spread):
    if sd > 0:
        return (spread - mu)/sd
    else:
        return 0

def update_window(stockA, stockB, spread):
    if (stockA, stockB) not in historical:
        window = deque(maxlen=window_size)
        historical[(stockA, stockB)] = (window,0,0)
        
    window,sum_x,sum_x2 = historical[(stockA,stockB)]
    if len(window) == window.maxlen:
        old = window.popleft()
        sum_x -= old
        sum_x2 -= old**2
    window.append(spread)
    
    sum_x += spread
    sum_x2 += spread**2
    
    n = len(window)
    mu = sum_x / n
    sd = np.sqrt((sum_x2/n - mu**2)* n/(n-1)) if n>1 else 0 

    historical[(stockA,stockB)] = (window,sum_x,sum_x2)


def act_on_it(stockA,stockB,z):
    key = (stockA,stockB)
    pos = prev_position.get(key, {'side': None, 'entry_z': None})
    
    if pos['side'] is None:
        if z > ENTRY_Z:
            pos['side'] = 'short'
            pos['entry_z'] = z
        elif z < -ENTRY_Z:
            pos['side'] = 'long'
            pos['entry_z'] = z
    
    elif pos['side'] == 'short' and z < EXIT_Z:
        pos['side'] = None
    elif pos['side'] == 'long' and z > -EXIT_Z:
        pos['side'] = None
    
    prev_position[key] = pos
    
##          By the end of the day
## Day 1: 4
## Day 2: 5   !!!window=[]!!!
## Day 3: 3   !!!window=[4], check 5 w/ window, get zX, take posX, add 5 to window!!!
## Day 4: 6   d(spread)=3-5, PnL=posX*d(spread), !!!window=[4,5], check 3 w/ window, take posY, add 3 to window!!!
## Day 5: 8   d(spread)=6-3, PnL=posY*d(spread), !!!window=[4,5,3], check 6 w/ window, take posZ, add 6 to window!!!
##

def run_pairs(prices, pairs, initial_capital):
    capital_per_pair = initial_capital / len(pairs)
    
    for i in range(2,prices.shape[0]):
        for stockA, stockB in pairs:
            hedge = hedge_dict[(stockA, stockB)]
            priceA_prev = prices.iloc[i-1][stockA]
            priceB_prev = prices.iloc[i-1][stockB]
            priceA_now  = prices.iloc[i][stockA]
            priceB_now  = prices.iloc[i][stockB]

            spread_prev = priceA_prev - hedge * priceB_prev
            spread_now = priceA_now - hedge * priceB_now
        
            old_side = {'long': 1, 'short': -1}.get(
                prev_position.get((stockA, stockB), {}).get('side'),
                0
            )
            notional = priceA_prev + abs(hedge)*priceB_prev
            pnl = ((old_side * (spread_now - spread_prev)) / notional) * capital_per_pair
        
            if (stockA, stockB) in historical:
                window, sum_x, sum_x2 = historical[(stockA, stockB)]
                n = len(window)
                if n > 1:
                    mu = sum_x / n
                    sd = np.sqrt((sum_x2/n - mu**2)* n/(n-1))
                else:
                    mu, sd = 0, 0
            else:
                mu, sd = 0, 0    
            z = check_spread_deviation(mu, sd, spread_prev)
    
            act_on_it(stockA, stockB, z)
        
            new_side = {'long': 1, 'short': -1}.get(
                prev_position.get((stockA, stockB), {}).get('side'),
                0
            )
        
            if new_side != old_side:
                cost = (SLIPPAGE_RATE + TRANSACTION_COST) * capital_per_pair
                pnl -= cost
        
            update_window(stockA, stockB, spread_prev)

            returns_list.append({
                'date': prices.index[i],
                'pair': (stockA, stockB),
                'pnl': pnl,
                'side': old_side
            })
    return returns_list

def run():
    start_date = input('Input start date (YYYY-MM-DD): ')
    end_date = input('Input end date (YYYY-MM-DD): ')
    prices,corr = pull_data(start_date,end_date)
    pairs = find_pairs(prices,corr)
    
    initial_capital = float(input('Input initial invested capital: '))
    returns_list = run_pairs(prices, pairs, initial_capital)
    
    returns_df = pd.DataFrame(returns_list)
    daily_pnl = returns_df.groupby('date')['pnl'].sum()
    print(daily_pnl.mean())
    print(daily_pnl.cumsum().plot(title="Cumulative mean PnL"))
    plt.show()
    
    
if __name__ == '__main__':
    run()
