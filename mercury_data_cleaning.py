import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# load saved csv
df_all = pd.read_csv('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/mercury_data.csv')
# drop useless column
df_all = df_all.drop('Unnamed: 0',axis=1)
# convert year + day into datetime format
df_all['date'] = pd.to_datetime(df_all['year'].astype(int).astype(str) + df_all['day'].astype(int).astype(str),format='%Y%j')
# rearrange columns
high_cadance_columns = [
    'date','year', 'day', 'dechr', 'hour', 'min', 'sec', 'rh', 'esh', 'clong', 'clat',
    'HGIlong', 'br', 'bt', 'bn', 'vp1r', 'vp1t', 'vp1n', 'crot', 'np1', 'vp1',
    'Tp1', 'vaz', 'vel', 'Bx', 'By', 'Bz', 'sBx', 'sBy', 'sBz', 'nal', 'val',
    'Tal', 'np2', 'vp2'
]
df_all = pd.DataFrame(df_all,columns=high_cadance_columns)
# filter using rh conditions again
df_all = df_all[(df_all['rh'] < 0.47) & (df_all['rh'] > 0.31)]
# print(df_all)
# df_all.to_csv('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/mercury_data_clean.csv')

# visuallise data
plt.figure()
plt.plot(df_all['rh'])
plt.show()
