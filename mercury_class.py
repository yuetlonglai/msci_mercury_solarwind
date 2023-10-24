import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.fft as fft
from scipy import signal 
from scipy import interpolate

class mercury_sw:
    def __init__(self,datapath) -> None:
        self._df = pd.read_csv(datapath)

        # some more cleaning
        try:
            self_df = self_df.drop('Unnamed: 0', axis=1)
        except:
            self_df = self_df
        
        self_df['datetime'] = pd.to_datetime(self_df['datetime'])
        self_df['Pram'] = 1.67e-27 * self_df['np1'] * 100**3 * (self_df['vp1'] * 1e3)**2
        self_df['absB'] = (self_df['Bx']**2 + self_df['By']**2 + self_df['Bz']**2)**0.5
        self_df = self_df[(self_df['Pram'] != 0)]

        pass

    def select_time(self,begin,end):
        self.begintime = begin
        self.endtime = end
        self._df = self._df[(self._df['datetime'] >= begin) & (self._df['datetime'] <= end)]
        return self
    
    def plot_pram_bfield(self, whichhelios):
        plt.figure(figsize=(10,8))
        plt.subplot(2,1,1)
        if whichhelios == 1:
            plt.title('Helios 1, Time: '+ self.begintime + ' - ' + self.endtime)
        elif whichhelios == 2:
            plt.title('Helios 2, Time: '+ self.begintime + ' - ' + self.endtime)
        else:
            plt.title('Time: '+ self.begintime + ' - ' + self.endtime)
        ax = plt.gca()
        plt.ylabel(r'$P_{ram} (Pa)$')
        # plt.xlabel('Time')
        plt.plot(self._df['datetime'].dt.strftime('%D %H:%M'), self._df['Pram'],'-',color='black')
        # plt.plot(pd.to_datetime(self._df['datetime']).dt.strftime('%D %H:%M'),signal.savgol_filter(df['Pram'],window_length=701,polyorder=3),'-',color='grey')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.subplot(2,1,2)
        ax = plt.gca()
        plt.ylabel('B (nT)')
        plt.xlabel('Time')
        plt.plot(self._df['datetime'].dt.strftime('%D %H:%M'), self._df['Bx'],'-',color='royalblue',label=r'$B_x$')
        plt.plot(self._df['datetime'].dt.strftime('%D %H:%M'), self._df['By'],'-',color='limegreen',label=r'$B_y$')
        plt.plot(self._df['datetime'].dt.strftime('%D %H:%M'), self._df['Bz'],'-',color='salmon',label=r'$B_z$')
        plt.plot(self._df['datetime'].dt.strftime('%D %H:%M'), self._df['absB'],'-',color='grey',label=r'$|B|$')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.legend()
        plt.show()
    
    def low_pass_pram(self,plotting=True):
        # first, interpolate the Pram data using Cubic Spline
        time_diff = self._df['datetime'].iloc[-1]-self._df['datetime'].iloc[0]
        t = np.linspace(0,time_diff.total_seconds(),len(self._df))
        print(time_diff.total_seconds())
        t_new = np.linspace(0,time_diff.total_seconds(),len(self._df)*3)
        Pram_cs = interpolate.CubicSpline(t,self._df['Pram'])(t_new)
        # then, perform low-pas filter on the interpolated data
        self.Pram_lp = signal.savgol_filter(Pram_cs,window_length=215,polyorder=3)
        # plot the result
        if plotting == True:
            plt.figure(figsize=(10,5))
            plt.title('Helios 2, Time: '+ self.begintime + ' - ' + self.endtime)
            plt.ylabel(r'$P_{ram} (Pa)$')
            plt.xlabel('Time (s)')
            plt.plot(t,self._df['Pram'],'-',color='Black',label='Original')
            plt.plot(t,self.Pram_lp,'--',color='cyan',label = 'Low-Pass')
            plt.legend(loc='best')
            plt.show()

        return self
    
    

