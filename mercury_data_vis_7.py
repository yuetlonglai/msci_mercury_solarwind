import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy import stats
from scipy import optimize
import networkx as nx
from sklearn import cluster

# load data
craft_num = 1
def prelim_data(df):
    # some more cleaning
    df = df.drop('Unnamed: 0', axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Pram'] = 1.67e-27 * df['np1'] * 100**3 * (df['vp1'] * 1e3)**2
    df['absB'] = (df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
    df = df[(df['Pram'] != 0)]
    # select time interval
    begin = '1977-04-01 00:00:00'
    end = '1977-04-10 23:59:59'
    df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
    df_select.sort_values('datetime',inplace=True)
    # print(df_select)
    return df_select

if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)

df1_select = prelim_data(pd.read_csv(f'mercury_data_1_clean.csv'))
df2_select = prelim_data(pd.read_csv(f'mercury_data_2_clean.csv'))


plotvariable = 'np1'
plt.figure()
plt.subplot(2,1,1)
ax = plt.gca()
plt.title('')
plt.xlabel('Datetime')
plt.ylabel(plotvariable)
plt.plot(df1_select['datetime'],df1_select['np1'],'-',color='royalblue',label='Helios 1')
plt.plot(df2_select['datetime'],df2_select['np1'],'-',color='salmon',label='Helios 2')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.legend()
plt.subplot(2,1,2)
ax = plt.gca()
plt.title('')
plt.xlabel('Datetime')
plt.ylabel('clong')
plt.plot(df1_select['datetime'],df1_select['clong'],'o',color='royalblue',label='Helios 1')
plt.plot(df2_select['datetime'],df2_select['clong'],'o',color='salmon',label='Helios 2')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.show()