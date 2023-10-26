import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.fft as fft
from scipy import signal
from scipy import signal 
from scipy import interpolate


# load data
craft_num = 0
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


# CME Event
v_th = 500 #km/s
n_th = 100   #cm-3
P_th = 1.67e-27 * n_th * 100**3 * (v_th* 1e3)**2

cme_df = df[df['Pram'] > P_th]

df['date'] = df['datetime'].dt.date

# Calculate daily mean of 'Pram' and store it in a new DataFrame
daily_mean = df.groupby('date')['Pram'].mean().reset_index()
daily_mean['datetime'] = pd.to_datetime(daily_mean['date'])  # Convert 'date' back to datetime

# cme_df['date'] = cme_df['datetime'].dt.date
# cme_df_mean = cme_df.groupby('date')['Pram'].mean().reset_index()
# cme_df_mean['datetime'] = pd.to_datetime(cme_df_mean['date'])

daily_mean_stats = daily_mean['Pram'].describe() #0:count,1:mean,2:std
# print(daily_mean_stats[1]+2*daily_mean_stats[2])

plt.figure(figsize=(8,6))
ax = plt.gca()
plt.ylabel(r'$P_{ram}$')
plt.xlabel('Datetime')
plt.plot(daily_mean['datetime'],daily_mean['Pram'],'.',color='blue',label='Daily mean '+r'$P_{ram}$')
# plt.plot(cme_df_mean['datetime'], cme_df_mean['Pram'],'x',color='red', label = 'CME Events')
plt.hlines(y = daily_mean_stats[1]+2*daily_mean_stats[2],xmin=min(daily_mean['datetime']),xmax=max(daily_mean['datetime']),colors='grey',linestyles='dashed',label=r'$\overline{P}_{ram}+2\sigma$')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.legend(loc='best')
plt.show()

cme_dates = daily_mean[(daily_mean['Pram'] >= daily_mean_stats[1]+2*daily_mean_stats[2])]
print(cme_dates['date'])


for i in cme_dates['date']:
    try:
        begin = str(i)+' 00:00:00'
        end = str(i)+' 23:59:59'
        df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
        # first, interpolate the Pram data using Cubic Spline
        time_diff = df_select['datetime'].iloc[-1]-df_select['datetime'].iloc[0]
        t = np.linspace(0,time_diff.total_seconds(),len(df_select))
        # print(time_diff.total_seconds())
        t_new = np.linspace(0,time_diff.total_seconds(),len(df_select)*3)
        Pram_cs = interpolate.CubicSpline(t,df_select['Pram'])(t_new)
        # then, perform low-pas filter on the interpolated data
        # Pram_lp = signal.savgol_filter(Pram_cs,window_length=105,polyorder=3)
        # plot the result
        # plt.figure(figsize=(10,5))
        plt.figure(figsize=(10,6))
        plt.title('Helios 2, Time: '+ begin + ' - ' + end)
        plt.ylabel(r'$P_{ram} (Pa)$')
        plt.xlabel('Time (s)')
        plt.plot(t,df_select['Pram'],'-',color='Black',label='Original')
        # plt.plot(t_new,Pram_lp,'--',color='cyan',label = 'Low-Pass')
        # plt.hlines(xmax=max(t),xmin=min(t),y=daily_mean_stats[1]+2*daily_mean_stats[2],color='grey',linestyles='--',label=r'$\overline{P}_{ram}+2\sigma$')
        # plt.hlines(xmax=max(t),xmin=min(t),y=df_select['Pram'].mean(),color='brown',linestyles='--',label=r'$\overline{P}_{ram}$')
        plt.legend(loc='upper right')
        plt.show()
    except:
        print('Error Date = '+str(i)+', Time Length = '+str(time_diff.total_seconds()))

