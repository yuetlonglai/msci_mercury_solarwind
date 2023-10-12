import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/gordonlai/Documents/ICL/ICL_Y4/MSci_Mercury/msci_mercury_solarwind/mercury_data_2_clean.csv')
df = df.drop('Unnamed: 0', axis=1)

df_select = df[(df['datetime'] >= '1977-05-03 13:05:31') & (df['datetime'] <= '1977-05-03 21:28:38')]
print(df_select)

Pram = 1.67e-27 * df_select['np1'] * 100**3 * (df_select['vp1'] * 1e3)**2

plt.figure(figsize=(10,6))
plt.plot(df_select['datetime'], Pram,'-',color='black')
plt.show()