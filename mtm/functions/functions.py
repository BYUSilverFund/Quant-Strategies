import modin.pandas as md
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from copy import deepcopy
set_matplotlib_formats('svg')

def import_clean_data(path):
    '''
    import and clean up the crsp_monthly csv file
    '''
    print('importing and cleaning data...')
    df = pd.read_csv(path)
    # convert dates to integers
    df['date'] = df['date'].astype(int)
    # convert dates to datetime
    df['date'] = pd.to_datetime(df['date'],format='%Y%m%d') 
    # convert rets to floats, set non-numeric rets to nans                     
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')  
    # compute market capitalization                                                                  
    df['MKTCAP'] = np.abs(df['PRC']) * df['SHROUT'] * 1000
    # sort data by stock, date
    df = df.sort_values(by=['PERMNO','date'],ascending=True)
    # set tickers to strings
    df['TICKER'] = df['TICKER'].astype(str)
    return df

def compute_price_volume(df):
    '''
    compute average daily trading volume
    '''
    print('computing average daily volumes...')
    df['ADV'] = (np.abs(df['PRC']) * df['VOL'])/21
    return df

def compute_mom_signal(df,window,lag):
    '''
    compute momentum signal
    '''
    print(f'computing momentum signal ({window}-month window and {lag}-month lag)')
    frames = []
    for name, frame in df.groupby('PERMNO'):
        frame['MOM'] = frame['RET'].rolling(window=window,min_periods=window).mean().shift(lag)
        frames.append(frame)
    df = pd.concat(frames)
    return df

def compute_reduced_dataset_adj(df,quantiles,top_mktcap=True,no_companies_per_quantile=500,reduce=True):
    frames = []
    if reduce == False:
        if top_mktcap == True:
            print(f'Taking top {no_companies_per_quantile} companies per quantile...')
        else:
            print(f'Taking all companies...')
        for name, frame in df.groupby(['date','SIZE','QTLS_adj']):
            if top_mktcap == True:
                frame = frame.sort_values(by=['MKTCAP']).head(no_companies_per_quantile)
                frames.append(frame)
            else:
                frames.append(frame)
        df = pd.concat(frames)
        return df
    else:
        if top_mktcap == True:
            print(f'reducing data to just top and bottom quantiles, taking top {no_companies_per_quantile} companies per quantile...')
        else:
            print(f'reducing data to just top and bottom quantiles, taking all companies...')
        for name, frame in df.groupby(['date','SIZE','QTLS_adj']):
            if name[2] == 1 or name[2] == quantiles:
                if top_mktcap == True:
                    frame = frame.sort_values(by=['MKTCAP']).head(no_companies_per_quantile)
                    frames.append(frame)
                else:
                    frames.append(frame)
            else:
                continue
        df = pd.concat(frames)
        return df
            
def compute_holding_periods_adj(df,hold_months):
    print('computing one-month holding periods...')
    df = df.set_index(['PERMNO','date'])
    df['FRM_DATE'] = df.index.get_level_values(level='date')
    df['HLD_BEGIN'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthBegin(1)
    df['HLD_END'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthEnd(hold_months)
    df = deepcopy(df[['FRM_DATE','QTLS','HLD_BEGIN','HLD_END','MKTCAP','MOM','SIZE','QTLS_adj']])
    return df

def merge_holding_periods_on_returns_adj(data,holding_periods,quantiles):
    print('merging return and holding period data...')
    portfolio = pd.merge(left=(data[['PERMNO','date','RET']]),right=(holding_periods),on=['PERMNO'],how='inner')
    portfolio = portfolio[(portfolio['HLD_BEGIN'] <= portfolio['date']) & (portfolio['date'] <= portfolio['HLD_END'])]
    portfolio = portfolio[['PERMNO','FRM_DATE','QTLS','HLD_BEGIN','HLD_END','date','RET','MKTCAP','MOM','SIZE','QTLS_adj']]
    port = portfolio.groupby(['date','SIZE','QTLS_adj','FRM_DATE']).mean().reset_index(level=[1,2,3])   # reset all indices except date
    small_low = port[(port.SIZE == 1) & (port['QTLS_adj'] == 1)]           # small low
    big_low = port[(port.SIZE == 2) & (port['QTLS_adj'] == 1)]             # big low
    small_high = port[(port.SIZE == 1) & (port['QTLS_adj'] == quantiles)]  # small high
    big_high = port[(port.SIZE == 2) & (port['QTLS_adj'] == quantiles)]    # big high
    size_adjusted_strategy = pd.concat([small_low.RET,big_low.RET,small_high.RET,big_high.RET],axis=1)
    size_adjusted_strategy.columns = ['small_low','big_low','small_high','big_high']
    size_adjusted_strategy['high'] = 0.5 * size_adjusted_strategy['small_high'] + 0.5 * size_adjusted_strategy['big_high']
    size_adjusted_strategy['low'] = 0.5 * size_adjusted_strategy['small_low'] + 0.5 * size_adjusted_strategy['big_low']
    size_adjusted_strategy['high-low'] = size_adjusted_strategy.high - size_adjusted_strategy.low
    size_adjusted_strategy['long_only_hi_adj'] = np.log(1 + size_adjusted_strategy['high']).cumsum()
    size_adjusted_strategy['long_only_lo_adj'] = np.log(1 + size_adjusted_strategy['low']).cumsum()
    size_adjusted_strategy['long_short_adj'] = np.log(1 + size_adjusted_strategy['high-low']).cumsum()
    return size_adjusted_strategy

def compute_reduced_dataset(df,quantiles,top_mktcap=True,no_companies_per_quantile=500,reduce=True):
    frames = []
    if reduce == False:
        if top_mktcap == True:
            print(f'Taking top {no_companies_per_quantile} companies per quantile...')
        else:
            print(f'Taking all companies...')
        for name, frame in df.groupby(['date','QTLS']):
            if top_mktcap == True:
                frame = frame.sort_values(by=['MKTCAP']).head(no_companies_per_quantile)
                frames.append(frame)
            else:
                frames.append(frame)
        df = pd.concat(frames)
        return df
    else:
        if top_mktcap == True:
            print(f'reducing data to just top and bottom quantiles, taking top {no_companies_per_quantile} companies per quantile...')
        else:
            print(f'reducing data to just top and bottom quantiles, taking all companies...')
        for name, frame in df.groupby(['date','QTLS']):
            if name[1] == 1 or name[1] == quantiles:
                if top_mktcap == True:
                    frame = frame.sort_values(by=['MKTCAP']).head(no_companies_per_quantile)
                    frames.append(frame)
                else:
                    frames.append(frame)
            else:
                continue
        df = pd.concat(frames)
        return df
            
def compute_holding_periods(df,hold_months):
    print('computing one-month holding periods...')
    df = df.set_index(['PERMNO','date'])
    df['FRM_DATE'] = df.index.get_level_values(level='date')
    df['HLD_BEGIN'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthBegin(1)
    df['HLD_END'] = df.index.get_level_values(level='date') + MonthEnd(0) + MonthEnd(hold_months)
    df = deepcopy(df[['FRM_DATE','QTLS','HLD_BEGIN','HLD_END','MKTCAP','MOM']])
    return df

def merge_holding_periods_on_returns(data,holding_periods):
    print('merging return and holding period data...')
    portfolio = pd.merge(left=(data[['PERMNO','date','RET']]),right=(holding_periods),on=['PERMNO'],how='inner')
    portfolio = portfolio[(portfolio['HLD_BEGIN'] <= portfolio['date']) & (portfolio['date'] <= portfolio['HLD_END'])]
    portfolio = portfolio[['PERMNO','FRM_DATE','QTLS','HLD_BEGIN','HLD_END','date','RET','MKTCAP','MOM']]
    port = portfolio.groupby(['date','QTLS','FRM_DATE'])[['RET']].mean().reset_index()
    port = port.groupby(['date','QTLS']).mean()
    return port.reset_index(), portfolio

    