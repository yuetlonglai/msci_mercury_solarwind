import pandas as pd
import numpy as np

df1 = pd.read_csv('mercury_data_1_clean.csv')
df2 = pd.read_csv('mercury_data_2_clean.csv')

df1['helios'] = 1
df2['helios'] = 2

df_tot = pd.concat([df1,df2])
df_tot = df_tot.drop('Unnamed: 0',axis=1)
print(df_tot)
df_tot.to_csv('mercury_data_0_clean.csv')