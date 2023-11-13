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

# histogram
# select year (for likelihood)
df = df[(df['year'] == 1975) & (df['rh'] <= 0.33)]
# low_thresh = 1e-9
# high_thresh = 7e-8
low_thresh = df['Pram'].mean() - 1.5 * df['Pram'].std()
high_thresh = df['Pram'].mean() + 2 * df['Pram'].std()
# low_thresh = np.percentile(df['Pram'],25) - 1.5 * (np.percentile(df['Pram'],75)-np.percentile(df['Pram'],25))
# high_thresh = np.percentile(df['Pram'],25) + 2 * (np.percentile(df['Pram'],75)-np.percentile(df['Pram'],25))
print(low_thresh,high_thresh)
'''
plt.figure()
# plt.title(f'Helios {craft_num}')
plt.title('Helios 1 & 2')
plt.ylabel('Count Density')
plt.xlabel(r'$P_{ram}$' +' (Pa)' )
plt.hist(df['Pram'], bins=500, density=True ,alpha=0.7, color='blue')
plt.vlines(x=low_thresh, ymin=0, ymax=1e8, linestyles='dashed', color='black')
plt.vlines(x=high_thresh, ymin=0, ymax=1e8, linestyles='dashed', color='black')
plt.xscale('log')
# plt.ylim(0,1e8)
plt.show()
'''
# percentage likelihood calculation
likelihood = [
    len(df[(df['Pram'] <= low_thresh) | (df['Pram'] >= high_thresh)])/len(df),
    len(df[(df['Pram'] <= low_thresh)])/len(df),
    len(df[(df['Pram'] >= high_thresh)])/len(df)
]
print('Percentage Likelihood for: \n Unusually low events = %.3f %% \n Unusually high events = %.3f %% \n Unusual events = %.3f %%' %(100*likelihood[1],100*likelihood[2],100*likelihood[0]))
'''
def skewed_gaussian_pdf(x, loc, scale, alpha,A):
    z = (x - loc) / scale
    pdf = 2 * A/scale * stats.norm.pdf(z) * stats.norm.cdf(alpha * z)
    return pdf

mean_max = 1.744521e-8
std_dev_max = 4.89044e-9 
initial_loc_min = mean_max  # Initial location (center) estimate
initial_scale_min = std_dev_max  # Initial scale (standard deviation) estimate
initial_alpha_min = -0.11  # Initial skewness estimate
A = 1500

initial_loc_max = mean_max # Initial location (center) estimate
initial_scale_max = std_dev_max  # Initial scale (standard deviation) estimate
initial_alpha_max = -0.11  # Initial skewness estimate

means_list_min = []
std_list_min = []
means_list_max = []
std_list_max = []
if craft_num == 2:
    year_range = range(1976,1979+1)
else:
    year_range = range(1975,1981+1)
# change in histograms over time
for i in year_range:
    plt.figure(figsize=(8,6))
    plt.title(f'Helios 1 at Year {i}')
    plt.ylabel('Count')
    # plt.xlabel(r'$P_{ram}$' +' (Pa)')
    counts_min, bin_edges_min, _ = plt.hist(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram'] ,bins=500, density=True ,alpha=0.7, color='blue', label='Perihelion')#, weights=np.ones_like(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram']) / len(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram']))
    counts_max, bin_edges_max, _  = plt.hist(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram'] ,bins=500, density=True ,alpha=0.7, color='red', label='Aphelion')#,weights=np.ones_like(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram']) / len(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram']))
    params_min, _ = optimize.curve_fit(skewed_gaussian_pdf, bin_edges_min[:-1], counts_min, p0=[initial_loc_min, initial_scale_min, initial_alpha_min,A])
    params_max, _ = optimize.curve_fit(skewed_gaussian_pdf, bin_edges_max[:-1], counts_max, p0=[initial_loc_max, initial_scale_max, initial_alpha_max,A])
    loc_min, scale_min, alpha_min, A_min = params_min
    loc_max, scale_max, alpha_max, A_max = params_max
    pdf_min = skewed_gaussian_pdf(bin_edges_min[:-1], loc_min, scale_min, alpha_min, A_min)
    pdf_max = skewed_gaussian_pdf(bin_edges_max[:-1], loc_max, scale_max, alpha_max, A_max)
    plt.plot(bin_edges_min[:-1], pdf_min, label='Skewed Gaussian Fit (Perihelion)', color='cyan')
    plt.plot(bin_edges_max[:-1], pdf_max, label='Skewed Gaussian Fit (Aphelion)', color='red')
    # plt.xscale('log')
    # plt.ylim(0,500)
    # plt.xlim(0.5e-9,2e-7)
    plt.legend()
    # plt.show()
    # print('Length of Perihelion  = %.3f' %(len(df[(df['year'] == i) & (df['rh'] <= 0.33)]['Pram'])))
    # print('Length of Aphelion  = %.3f' %(len(df[(df['year'] == i) & (df['rh'] >= 0.44)]['Pram'])))
    means_list_min.append(loc_min)
    std_list_min.append(scale_min)
    means_list_max.append(loc_max)
    std_list_max.append(scale_max)

plt.figure(figsize=(10,5))
plt.suptitle(f'Helios {craft_num}')
plt.subplot(1,2,1)
plt.grid()
plt.ylabel(r'$\overline{P}_{ram}$')
plt.xlabel('Year')
plt.plot(year_range,means_list_min,'-o',color='blue',label='Perihelion')
plt.plot(year_range,means_list_max,'-o',color='red',label='Aphelion')
plt.ylim(0,1.7e-8)
plt.legend()
plt.subplot(1,2,2)
plt.grid()
plt.xlabel('Year')
plt.ylabel(r'$\sigma(\overline{P}_{ram})$')
plt.plot(year_range,std_list_min,'-o',color='blue',label='Perihelion')
plt.plot(year_range,std_list_max,'-o',color='red',label='Aphelion')
plt.ylim(0,1.7e-8)
plt.legend()
# plt.show()
'''

'''
df['year-month'] = df['datetime'].dt.strftime('%Y-%m')
df = df.sort_values(by='year-month')
plt.figure(figsize=(15,8))
plt.grid()
plt.title(f'Helios {craft_num}')
sns.boxplot(data=df, x='year-month', y='Pram')
plt.ylabel(r'$P_{ram}$')
plt.xticks(rotation=90)
plt.legend()
plt.show()
'''
