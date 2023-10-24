import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.fft as fft
from scipy import signal 
from scipy import interpolate

# load data
df = pd.read_csv('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/mercury_data_2_clean.csv')
# some more cleaning
df = df.drop('Unnamed: 0', axis=1)
df['datetime'] = pd.to_datetime(df['datetime'])
df['Pram'] = 1.67e-27 * df['np1'] * 100**3 * (df['vp1'] * 1e3)**2
df['absB'] = (df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
df = df[(df['Pram'] != 0)]
# select time interval
begin = '1977-05-03 00:00:00'
end = '1977-05-05 23:59:59'
df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
# print(df_select)

# visualise
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.title('Helios 2, Time: '+ begin + ' - ' + end)
ax = plt.gca()
plt.ylabel(r'$P_{ram} (Pa)$')
# plt.xlabel('Time')
plt.plot(df_select['datetime'].dt.strftime('%D %H:%M'), df_select['Pram'],'-',color='black')
# plt.plot(pd.to_datetime(df_select['datetime']).dt.strftime('%D %H:%M'),signal.savgol_filter(df['Pram'],window_length=701,polyorder=3),'-',color='grey')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.subplot(2,1,2)
ax = plt.gca()
plt.ylabel('B (nT)')
plt.xlabel('Time')
plt.plot(df_select['datetime'].dt.strftime('%D %H:%M'), df_select['Bx'],'-',color='royalblue',label=r'$B_x$')
plt.plot(df_select['datetime'].dt.strftime('%D %H:%M'), df_select['By'],'-',color='limegreen',label=r'$B_y$')
plt.plot(df_select['datetime'].dt.strftime('%D %H:%M'), df_select['Bz'],'-',color='salmon',label=r'$B_z$')
plt.plot(df_select['datetime'].dt.strftime('%D %H:%M'), df_select['absB'],'-',color='grey',label=r'$|B|$')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.legend()
plt.show()

# Interpolation
time_diff = df_select['datetime'].iloc[-1]-df_select['datetime'].iloc[0]
t = np.linspace(0,time_diff.total_seconds(),len(df_select))
print(time_diff.total_seconds())
t_new = np.linspace(0,time_diff.total_seconds(),len(df_select)*3)
Pram_cs = interpolate.CubicSpline(t,df_select['Pram'])(t_new)
# plt.figure()
# plt.plot(np.linspace(0,time_diff.total_seconds(),len(df_select)), df_select['Pram'],'-',color='black')
# plt.plot(t_new,Pram_cs,'--',color='blue')
# plt.show()

# Fourier Transform
time_diff = df_select['datetime'].iloc[-1]-df_select['datetime'].iloc[0]
yf = fft.rfft(np.array(Pram_cs))
xf = fft.rfftfreq(len(df_select)*3,40.5)
plt.figure()
# plt.plot(np.linspace(0,time_diff.total_seconds(),len(df_select)), df_select['Pram'],'-',color='black')
plt.plot(xf,yf)
plt.yscale('log')
plt.xscale('log')
plt.show()

# Low-pass filter
Pram_lp = signal.savgol_filter(Pram_cs,window_length=215,polyorder=3)
t = np.linspace(0,time_diff.total_seconds(),len(Pram_cs))
plt.figure(figsize=(10,5))
plt.title('Helios 2, Time: '+ begin + ' - ' + end)
plt.ylabel(r'$P_{ram} (Pa)$')
plt.xlabel('Time (s)')
# plt.plot(t_new,Pram_cs,'-',color='grey',label='Original')
plt.plot(np.linspace(0,time_diff.total_seconds(),len(df_select)),df_select['Pram'],'-',color='black',label='Original')
plt.plot(t,Pram_lp,'--',color='cyan',label = 'Low-Pass')
plt.legend()
plt.show()



# Histogram
# plt.figure()
# sns.histplot(data=df,x='Pram',bins=2000,color='blue',kde=True)
# plt.xlabel(r'$P_{ram} (Pa)$')
# plt.xlim(0,7e-8)
# plt.show()