import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy import stats
from scipy import optimize
from scipy import signal
from scipy import fftpack
from statsmodels.nonparametric import smoothers_lowess
from statsmodels.tsa.seasonal import MSTL, DecomposeResult
from astropy.timeseries import LombScargle
import statistics

craft_num = 1
def prelim_data(df,select=False):
    # some more cleaning
    df = df.drop('Unnamed: 0', axis=1)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Pram'] = 1.67e-27 * df['np1'] * 100**3 * (df['vp1'] * 1e3)**2
    df['absB'] = (df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
    df = df[(df['Pram'] != 0)]
    df['date'] = df['datetime'].dt.date
    df.sort_values('datetime',inplace=True)
    k_value = 50
    df['Rstdoff'] = 1+k_value * df['Pram']**(-1/6) / 2439.7 # in unit of radius of Mercury R_M
    return df

def time_select(df,b,e):
    # select time interval
    # begin = '1977-04-01 00:00:00'
    # end = '1977-04-10 23:59:59'
    begin=b
    end=e
    df_select = df[(df['datetime'] >= begin) & (df['datetime'] <= end)]
    df_select.sort_values('datetime',inplace=True)
    return df_select

def low_pass(timeseries, lowpassvar, f=0.010):
    # datetime_new = pd.array([timeseries['datetime'].iloc[0] + pd.to_timedelta(a, unit='s') for a in t_new])
    Pram_lp = smoothers_lowess.lowess(timeseries[lowpassvar],timeseries['datetime'], is_sorted=True, frac=f, it=0)
    return Pram_lp[:,1]

def datetime_to_seconds(datetime):
    # pandas
    return pd.to_timedelta(datetime - datetime.iloc[0]).dt.total_seconds()

def FT(timeseries,spacing=40.5):
    xf = fftpack.rfftfreq(len(timeseries),d=spacing)
    yf = fftpack.rfft(np.array(timeseries))
    # xf = fftpack.fftshift(xf)
    # yf = fftpack.fftshift(yf)
    return xf, yf

def Lomb_Scargle(timeseries,var,lowpass=False,y=[]):
    if lowpass == False:
        return LombScargle(datetime_to_seconds(timeseries['datetime']),timeseries[var]).autopower()
    else:
        return LombScargle(datetime_to_seconds(timeseries['datetime']),y).autopower()
    

def high_freq_change(timeseries, time, d=100):
    grads = abs(np.gradient(timeseries,datetime_to_seconds(time)))
    h = grads.mean() + 2* grads.std()
    peaks, _ = signal.find_peaks(grads,height=h,distance=d)
    return peaks
    
def continuous_periods_with_dist(df,thres = 5, leng = 1000):
    # df = df[(df['rh'] < 0.33)]  # If you want to filter based on a condition
    df = df.sort_values('datetime')
    df['time_diff'] = df['datetime'].diff()
    threshold = pd.Timedelta(minutes=thres)
    df['group'] = (df['time_diff'] >= threshold).cumsum()

    # dfs = [group_df.drop(['timediff', 'group'], axis=1) for group_df in df.groupby((df['time_diff'] >= threshold).cumsum()) if len(group_df) > 200]
    dfs = []
    for group_df in df.groupby((df['time_diff'] >= threshold).cumsum()):
        if len(group_df[1]) > leng:
            dfs.append(group_df[1])
    return sorted(dfs, key=lambda x:(x['datetime'].iloc[-1]-x['datetime'].iloc[0]),reverse=True)

def CDS(y,x): 
    # x has to be datetime format, x and y have to be in pandas 
    grad = []
    for i in range(len(x)):
        # print(i)
        if i == 0:
            g = (y[i+1]-y[i])/((x.iloc[i+1]-x.iloc[i]).total_seconds())
        else:
            g = (y[i]-y[i-1])/((x.iloc[i]-x.iloc[i-1]).total_seconds())
        # else:
        #     g = (y.iloc[i+1]-y.iloc[i-1])/((x.iloc[i+1]-x.iloc[i-1]).total_seconds())
        grad.append(g)
    return np.array(grad)

def time_variability(actual,var,ratio=False): # find how noisy
    # actual needs to be in pandas dataframe, lowpass is just array
    lowpass = low_pass(timeseries=actual,lowpassvar=var)
    if ratio == False:
        variability = statistics.median(abs((np.array(actual[var]) - np.array(lowpass))))
    else:
        variability = statistics.median(abs((np.array(lowpass) / np.array(actual[var]))))
    return variability

def large_changes_search(actual,var):
    lowpass = abs(CDS(low_pass(timeseries=actual,lowpassvar=var),actual['datetime']))
    threshold = np.mean(lowpass) + 2 * np.std(lowpass)
    lowpass_mask = list(lowpass > threshold)
    big_jump_count = lowpass_mask.count(True)
    return big_jump_count

def trend_decomposition(timeseries,var):
    mstl = MSTL(timeseries[var],periods=[24,24*7])
    res = mstl.fit()
    return res

def skin_depth(sigma,omega): # in km
    return np.sqrt(2/(sigma*omega*4e-7*np.pi))

def radial_conductivity_heyner(r):
    if r >= 0 and r <= 1740:
        return [1e5,1e7]
    elif r > 1740 and r <= 1940:
        return [1e2,1e3]
    elif r > 1940 and r <= 2040:
        return [10**-0.5,10**0.5]
    elif r > 2040 and r <= 2300:
        return [1e-3,10**0.7]
    elif r > 2300 and r <= 2400:
        return [1e-7,1e-2]
    else:
        print("Reached the planet's surface")
        return [0.0,0.0]
    
def skin_depth_total(omega,low_bound = False):
    layer_widths = [100,260,100,200,1740]
    radii = [2400,2300,2040,1940,1740]
    if low_bound == False:
        index = 1
    else:
        index = 0
    depths = []
    for i in range(len(radii)):
        depth = skin_depth(radial_conductivity_heyner(radii[i])[index],omega=omega) 
        if depth > layer_widths[i]:
            depths.append(layer_widths[i])
        else:
            depths.append(depth)
            break
    return sum(depths)

def attenuation_total(omega,power,low_bound=False,alldata=False):
    # layer_widths = [100,260,100,200,1740]
    radii = np.array([2400,2300,2040,1940,1740,0])
    layer_depth = 2400-radii
    if low_bound == False:
        index = 1
    else:
        index = 0
    threshold = 0.2 * 1e-1
    # threshold = 0.2 * power
    depths = 0
    last_atten = [power]
    all_z = []
    all_atten = []
    for i in range(len(radii)-1):
        z = np.linspace(layer_depth[i],layer_depth[i+1],1000)
        depth = skin_depth(radial_conductivity_heyner(radii[i])[index],omega=omega) 
        atten = last_atten[-1] * np.exp(-z/depth)
        zz = z[0:-2]
        all_z = all_z + list(zz)
        all_atten = all_atten + list(last_atten[-1] * np.exp(-zz/depth))
        if atten[-1] >= threshold:
            depths = layer_depth[i]
            last_atten.append(atten[-1])
        else:
            for j in range(len(z)):
                if atten[j] < threshold:
                    depths = z[j]
                    break
            break
    if alldata == False:
        return depths
    else:
        return depths, all_z, all_atten


if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)

######################################################################################################################################
# code
# load data
df1 = prelim_data(pd.read_csv(f'mercury_data_1_clean.csv'))
df2 = prelim_data(pd.read_csv(f'mercury_data_2_clean.csv'))

begintime = '1979-05-20 00:00:00'
endtime = '1979-06-05 23:59:59'
df1_select = time_select(df1, begintime, endtime)
df1_select_lp = low_pass(df1_select, 'Pram')
'''
### plot the data and lowpass
plt.figure(figsize=(10,5))
ax = plt.gca()
plt.ylabel(r'$P_{ram} (Pa)$')
plt.xlabel('Time (s)')
plt.plot(df1_select['datetime'],df1_select['Pram'],'-',color='black',label='Original')
plt.plot(df1_select['datetime'],df1_select_lp,'-',color='cyan',label = 'Low-Pass')
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.legend()
# plt.show()


### plot the gradient of the data
df1_grad = CDS(np.array(df1_select['Pram']),df1_select['datetime'])
plt.figure(figsize=(10,5))
ax = plt.gca()
plt.ylabel(r'$\partial_t P_{ram} (Pa)$')
plt.xlabel('Time (s)')
plt.plot(df1_select['datetime'],df1_grad,'-',color='black',label='Original')
plt.plot(df1_select['datetime'],CDS(df1_select_lp,df1_select['datetime']),color='royalblue',label='Low-Pass')
# plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.legend()
# plt.show()
'''
##########################################
df1_continuous_select = continuous_periods_with_dist(df1,thres=10,leng=1000)
# df1_continuous_select = [
#     [pd.to_datetime('1978-04-12'),pd.to_datetime('1978-04-23')],
#     [pd.to_datetime('1979-05-25'),pd.to_datetime('1979-06-01')],
#     [pd.to_datetime('1977-04-01'),pd.to_datetime('1977-04-05')]
#     ]
# print(len(df1_continuous_select))
# begintime = '1979-05-20 00:00:00'
# endtime = '1979-06-05 23:59:59'
for i in range(len(df1_continuous_select)):
    num = i
    begintime = df1_continuous_select[num]['datetime'].iloc[0]
    endtime = df1_continuous_select[num]['datetime'].iloc[-1]
    # begintime = df1_continuous_select[num][0]
    # endtime = df1_continuous_select[num][1]
    df1_select = time_select(df1, begintime, endtime)
    df1_select_lp = low_pass(df1_select, 'Pram')
    df1_select_lp_2 = low_pass(df1_select, 'Pram',f=0.2)

    # FT 
    ft_df1 = Lomb_Scargle(df1_select,'Pram')
    ft_df1_lp = Lomb_Scargle(df1_select,'Pram',lowpass=True,y=df1_select_lp_2)

    # skin depth
    # peaks,_ = signal.find_peaks(ft_df1[1],height=0.5e-1,distance=10)
    # freq_dominant = ft_df1_lp[0][peaks][np.argmax(ft_df1_lp[1][peaks])]
    # skindepth = skin_depth_total(freq_dominant*2*np.pi)
    # skindepths = np.array([skin_depth_total(ft_df1[0][p]*2*np.pi) for p in peaks])
    skindepthboundary = np.array([skin_depth_total(a*2*np.pi) for a in np.logspace(-8,0,500)])
    signal_depth = np.array([attenuation_total(2*np.pi*ft_df1[0][a],ft_df1[1][a]) for a in range(len(ft_df1[0]))])

    # plot
    fig = plt.figure(figsize=(12,8))
    # plt.suptitle(r'$P_{ram}$ ' + 'and Lomb-Scargle Periodogram')
    fig.suptitle(str(begintime) +' - '+ str(endtime) +', Length = ' +str(endtime-begintime))# + f', skin depth = {skindepth:2f}' + ' (km)')
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1 = plt.gca()
    plt.ylabel(r'$P_{ram} (Pa)$')
    # plt.ylabel(r'$R_{stdoff}$' + r' $(R_M)$')
    plt.xlabel('Datetime')
    plt.plot(df1_select['datetime'],df1_select['Pram'],'-',color='black',label='Original')
    plt.plot(df1_select['datetime'],df1_select_lp,'-',color='cyan',label = 'Weak Low-Pass')
    plt.plot(df1_select['datetime'],df1_select_lp_2,'-',color='orange',label = 'Strong Low-Pass')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.legend()
    ax2 = plt.subplot2grid((2,2),(1,0))
    # plt.title('Lomb-Scargle Periodogram')
    plt.ylabel('Lomb-Scargle Power ')#() + r'$(Pa^2Hz^{-1})$')
    plt.xlabel('Frequency (Hz)')
    plt.plot(ft_df1[0],ft_df1[1],'-',color='black',label='Original')
    # plt.plot(ft_df1_lp[0],ft_df1_lp[1],'-',color='orange',label='Strong Low-pass')
    plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/40.5,linestyles='dashed',colors='grey',label='40.5 seconds')
    plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/3600,linestyles='dashed',colors='green',label='1 hour')
    plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/86400/(4/24),linestyles='dashed',colors='deepskyblue',label='4 hours')
    plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/86400,linestyles='dashed',colors='darkblue',label='1 day')
    # plt.plot(freq_dominant,max(ft_df1_lp[1][:100]),'.',color='red')
    # plt.plot(ft_df1[0][peaks],ft_df1[1][peaks],'.',color='red')
    # plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/86400,linestyles='dashed',colors='red',label='1 day')
    # plt.vlines(ymin=min(ft_df1[1]),ymax=max(ft_df1[1]),x=1/86400/7,linestyles='dashed',colors='blue',label='7 day')
    plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    ax3 = plt.subplot2grid((2,2),(0,1),rowspan=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Radius (km)')
    plt.fill_between(y1=2400,y2=2300,x=np.linspace(0,max(ft_df1[0]),100),color='grey',label='Crust',alpha=0.7)
    plt.fill_between(y1=2300,y2=2040,x=np.linspace(0,max(ft_df1[0]),100),color='brown',label='Mantle',alpha=0.95)
    plt.fill_between(y1=2040,y2=1940,x=np.linspace(0,max(ft_df1[0]),100),color='brown',alpha=0.93)
    plt.fill_between(y1=1940,y2=1740,x=np.linspace(0,max(ft_df1[0]),100),color='brown',alpha=0.91)
    plt.fill_between(y1=1740,y2=1440,x=np.linspace(0,max(ft_df1[0]),100),color='red',label='Core',alpha=0.7)
    plt.plot(np.logspace(-8,0,500),2400-skindepthboundary,'--',color='royalblue',alpha=0.7,label='Skin Depth')
    plt.plot(ft_df1[0],2400-signal_depth,'-',color='black',label='Signal Depth')
    # for p in range(len(peaks)):
    #     plt.vlines(x=ft_df1[0][peaks[p]],ymax=2400,ymin=2400-skindepths[p],color='black',linestyles='solid')
    plt.xscale('log')
    plt.ylim(1440,2450)
    plt.xlim(min(ft_df1[0]),max(ft_df1[0]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

'''
### time variability plot
# df1_goodness = time_variability(df1_select,'Pram')
# print(df1_goodness)
sunspot_data = pd.read_csv('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/SN_m_tot_V2.0.csv',delimiter=';')
sunspot_data.columns = ['year','month','decimaldate','sunspots','others1','others2','others3']
# sunspot_data['datetime'] = pd.to_datetime(sunspot_data['decimaldate'], origin='unix').dt.strftime('%Y%M%D')
# print(sunspot_data)
sunspot_data_x = sunspot_data[(sunspot_data['decimaldate'] >= 1975)&(sunspot_data['decimaldate'] <= 1982)]['decimaldate']
sunspot_data_y = sunspot_data[(sunspot_data['decimaldate'] >= 1975)&(sunspot_data['decimaldate'] <= 1982)]['sunspots']
sunspot_data_y_smooth = smoothers_lowess.lowess(sunspot_data[(sunspot_data['decimaldate'] >= 1975)&(sunspot_data['decimaldate'] <= 1982)]['sunspots'],sunspot_data_x,is_sorted=True, frac=0.250, it=0)[:,1]
continous_df1_1 = continuous_periods_with_dist(df1[(df1['rh'] <= 0.33)])
continous_df1_2 = continuous_periods_with_dist(df1[(df1['rh'] >= 0.44)])

# print(len(continous_df1))
goodnesses_1 = []
for j in year_range:
    good = []
    num_in_year = 0
    for i in range(len(continous_df1_1)):
        if int(pd.to_datetime(continous_df1_1[i]['datetime'].mean()).year) == j:
            good.append(time_variability(continous_df1_1[i],'Pram'))
            num_in_year+=1
    # print(num_in_year)
    goodnesses_1.append(np.mean(good))
goodnesses_2 = []
for j in year_range:
    good = []
    num_in_year = 0
    for i in range(len(continous_df1_2)):
        if int(pd.to_datetime(continous_df1_2[i]['datetime'].mean()).year) == j:
            good.append(time_variability(continous_df1_2[i],'Pram'))
            num_in_year+=1
    # print(num_in_year)
    goodnesses_2.append(np.mean(good))
fig1,ax1 = plt.subplots()
# plt.title('')
ax1.set_xlabel('Year')
ylabel1color = 'royalblue'
ax1.set_ylabel(r'$\Delta P_{ram} (Pa)$',color='blue')
ax1.plot(np.array(year_range),goodnesses_1,'-o',color='darkblue',label='Perihelion Time Variability')
ax1.plot(np.array(year_range),goodnesses_2,'-o',color='royalblue',label='Aphelion Time Variability')
ax1.tick_params(axis='y', labelcolor=ylabel1color)
ax2 = ax1.twinx()
ylabel2color='green'
ax2.set_ylabel('Number of Sunspots',color=ylabel2color)
ax2.plot(sunspot_data_x,sunspot_data_y,'--',color='grey',label='Sunspot Number')
ax2.plot(sunspot_data_x,sunspot_data_y_smooth,'-',color=ylabel2color,label='Sunspot Number (Smoothed)')
ax2.tick_params(axis='y', labelcolor=ylabel2color)
fig1.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
plt.show()
'''
# fig,ax = plt.subplots()
# # plt.plot(df1_select['datetime'],trend_decomposition(df1_select,'Pram').trend)
# ax = trend_decomposition(df1_select,'Pram').plot()
# plt.tight_layout()
# plt.show()