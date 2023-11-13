import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
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
# print(df_select)

if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)

low_percentile = 0.05
high_percentile = 1-low_percentile

def spread_eval(data):
    # data must be a column in pandas dataframe
    median = data.median()
    top_5_percentile = data.quantile(high_percentile)
    bottom_5_percentile = data.quantile(low_percentile)
    alpha = median - bottom_5_percentile
    beta = top_5_percentile - median
    return [median,bottom_5_percentile,top_5_percentile,alpha,beta]

def plot_subplots_histogram(data,colour,year,den=True):
    # plt.grid()
    plt.xlabel(r'$P_{ram}$')
    plt.ylabel('Count Density')
    if colour == 'blue':
        bin_height,bin_edges,_ = plt.hist(data,bins=500,density=den,alpha=0.7,color=colour,label='Perihelion')
        print(str(year) + ' Perhelion:')
    elif colour == 'red':
        bin_height,bin_edges,_ = plt.hist(data,bins=500,density=den,alpha=0.7,color=colour,label='Aphelion')
        print(str(year) + ' Aphelion:')
    plt.vlines(x=spread_eval(data=data)[0],ymin=0,ymax=10e8,color='black',linestyles='dashed',label='Median')
    plt.vlines(x=spread_eval(data=data)[1],ymin=0,ymax=10e8,color='aqua',linestyles='dashed',label=f'{(low_percentile*100)}th Percentile')
    plt.vlines(x=spread_eval(data=data)[2],ymin=0,ymax=10e8,color='aquamarine',linestyles='dashed',label=f'{(high_percentile*100)}th Percentile')
    plt.legend()
    plt.xscale('log')
    # plt.xlim(1.5e-9,1e-7)
    plt.ylim(0,max(bin_height))
    print('Alpha = %.3e, Beta = %.3e, a+b = %.3e' %(spread_eval(data=data)[-2],spread_eval(data=data)[-1],spread_eval(data=data)[-2]+spread_eval(data=data)[-1]))

plotvariable = 'Pram'

plt.figure(figsize=(12,8))
plt.suptitle('Helios 1')
plt.subplot(2,2,1)
plt.title('1975')
plot_subplots_histogram(df[(df['year'] == 1975) & (df['rh'] <= 0.33)][plotvariable],colour='blue',year='1975')
plt.subplot(2,2,2)
plt.title('1981')
plot_subplots_histogram(df[(df['year'] == 1981) & (df['rh'] <= 0.33)][plotvariable],colour='blue',year='1981')
plt.subplot(2,2,3)
plot_subplots_histogram(df[(df['year'] == 1975) & (df['rh'] >= 0.44)][plotvariable],colour='red',year='1975')
plt.subplot(2,2,4)
plot_subplots_histogram(df[(df['year'] == 1981) & (df['rh'] >= 0.44)][plotvariable],colour='red',year='1981')
plt.show()

alphas_1=[]
betas_1=[]
alphas_2=[]
betas_2=[]
median_1=[]
median_2=[]
for i in year_range:
    per = spread_eval(df[(df['year'] == i) & (df['rh'] <= 0.33)][plotvariable])
    aph = spread_eval(df[(df['year'] == i) & (df['rh'] >= 0.44)][plotvariable])
    alphas_1.append(per[-2])
    betas_1.append(per[-1])
    alphas_2.append(aph[-2])
    betas_2.append(aph[-1])
    median_1.append(per[0])
    median_2.append(aph[0])

plt.figure(figsize=(8,6))
plt.grid()
# plt.title(f'{} and {} Percentile as threshold')
plt.xlabel('Year')
# plt.ylabel(r'$P_{ram}$')
plt.ylabel(plotvariable)
# plt.plot(year_range,alphas_1,'-o',color='deepskyblue',label='Perihelion Alpha')
# plt.plot(year_range,alphas_2,'-o',color='lightcoral', label='Aphelion Alpha')
# plt.plot(year_range,betas_1,'-o',color='royalblue',label='Perihelion Beta')
# plt.plot(year_range,betas_2,'-o',color='salmon',label='Aphelion Beta')
plt.plot(year_range,np.array(alphas_1)+np.array(betas_1),'-o',color='blue',label='Perihelion Total')
plt.plot(year_range,np.array(alphas_2)+np.array(betas_2),'-o',color='red',label='Aphelion Total')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.grid()
plt.title('Median Difference')
plt.xlabel('year')
plt.ylabel(r'$P_{ram}$')
plt.plot(year_range,np.array(median_1)-np.array(median_2),'-o',color='black')
plt.show()





