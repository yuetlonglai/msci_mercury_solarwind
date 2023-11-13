import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy import stats
from scipy import optimize


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

if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)
'''
# change in histograms over time
for i in year_range:
    plt.figure(figsize=(15,7))
    plt.suptitle(f'Helios {craft_num} at Year {i}')
    plt.subplot(1,3,1)
    plt.ylabel('Count Density')
    plt.xlabel(r'$P_{ram}$' +' (Pa)')
    plt.hist(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram'] ,bins=500, density=True ,alpha=0.7, color='blue', label='Perihelion')
    plt.hist(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram'] ,bins=500, density=True ,alpha=0.7, color='red', label='Aphelion')
    plt.xscale('log')
    plt.subplot(1,3,2)
    plt.xlabel('vp1 (km/s)')
    plt.hist(df[(df['year'] == i) & (df['rh'] <= 0.33)]['vp1'] ,bins=500, density=True ,alpha=0.7, color='blue', label='Perihelion')
    plt.hist(df[(df['year'] == i) & (df['rh'] >= 0.44)]['vp1'] ,bins=500, density=True ,alpha=0.7, color='red', label='Aphelion')
    plt.subplot(1,3,3)
    plt.xlabel('np1 (cm^-3)')
    plt.hist(df[(df['year'] == i) & (df['rh'] <= 0.33)]['np1'] ,bins=500, density=True ,alpha=0.7, color='blue', label='Perihelion')
    plt.hist(df[(df['year'] == i) & (df['rh'] >= 0.44)]['np1'] ,bins=500, density=True ,alpha=0.7, color='red', label='Aphelion')
    # plt.ylim(0,500)
    # plt.xlim(0.5e-9,2e-7)
    plt.legend()
    # plt.show()
    # print('Length of Perihelion  = %.3f' %(len(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram'])))
    # print('Length of Aphelion  = %.3f' %(len(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram'])))
'''


plotvariable='Pram'
plt.figure(figsize=(10,6))
plt.suptitle(f'Helios {craft_num}')
plt.ylabel('Count Density')
# plt.xlabel(r'$P_{ram}$' +' (Pa)')
plt.xlabel(plotvariable)
plt.hist(df[(df['year'] == 1975)][plotvariable] ,bins=500, density=False ,alpha=0.7, color='blue',label='1975')
plt.hist(df[(df['year'] == 1981)][plotvariable] ,bins=500, density=False ,alpha=0.7, color='red',label='1981')
# plt.vlines(x=df[(df['year'] == 1975)][plotvariable].mean(),ymin=0,ymax=1,color='blue',linestyles='dashed',label='1975 Mean')
# plt.vlines(x=df[(df['year'] == 1981)][plotvariable].mean(),ymin=0,ymax=1,color='red',linestyles='dashed',label='1981 Mean')
plt.xscale('log')
# plt.ylim(0,0.03)
plt.legend()
plt.show()
print('Percentage Difference: %.3f %%' %(100*(df[(df['year'] == 1981)]['Pram'].mean()-df[(df['year'] == 1975)]['Pram'].mean())/df[(df['year'] == 1975)]['Pram'].mean()))
