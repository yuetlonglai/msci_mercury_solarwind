import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy import stats
from scipy import optimize
import networkx as nx

# load data
craft_num = 1
df = pd.read_csv(f'mercury_data_{craft_num}_clean.csv')
# some more cleaning
df = df.drop('Unnamed: 0', axis=1)
df['datetime'] = pd.to_datetime(df['datetime'])
df['Pram'] = 1.67e-27 * df['np1'] * 100**3 * (df['vp1'] * 1e3)**2
df['absB'] = (df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
df = df[(df['Pram'] != 0)]
# select time interval
begin = '1977-05-03 00:00:00'
end = '1977-05-03 23:59:59'
df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
# print(df_select)

if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)

# KLD D distance
