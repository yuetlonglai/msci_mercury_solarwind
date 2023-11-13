import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy import stats
from scipy import optimize
import networkx as nx
from sklearn import cluster

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
# end of prelim

# Calculate daily mean of 'Pram' and store it in a new DataFrame
# df['date'] = df['datetime'].dt.date
# daily_mean = df.groupby('date')['Pram'].mean().reset_index()
# daily_mean['date'] = pd.to_datetime(daily_mean['date'])  # Convert 'date' back to datetime
# daily_mean['Days_Since_Given_Date'] = (daily_mean['date'] - daily_mean['date'].iloc[0]).dt.days
# daily_mean_df = pd.concat([daily_mean['Days_Since_Given_Date'],daily_mean['Pram']],axis=1)

# Calculate hourly mean of 'Pram' and store it in a new DataFrame
df['date-hour'] = df['datetime'].dt.strftime('%Y-%m-%d %H')
daily_mean = df.groupby('date-hour')['Pram'].mean().reset_index()
daily_mean['date-hour'] = pd.to_datetime(daily_mean['date-hour'])  # Convert 'date' back to datetime
daily_mean['Hours_Since_Given_Date'] = (daily_mean['date-hour'] - daily_mean['date-hour'].iloc[0]).dt.total_seconds()/3600
daily_mean_df = pd.concat([daily_mean['Hours_Since_Given_Date'],daily_mean['Pram']],axis=1)

# Cluster data
agg_clustering = cluster.AgglomerativeClustering(n_clusters=13,linkage='single')
daily_mean_df['cluster_label'] = agg_clustering.fit_predict(daily_mean_df)

plt.figure()
ax = plt.gca()
plt.scatter(daily_mean['date-hour'], daily_mean_df['Pram'], c=daily_mean_df['cluster_label'], marker='o', edgecolor='black', s=50, cmap='viridis')
plt.title('Agglomerative Clustering Results')
plt.xlabel('Datetime')
plt.ylabel('Pram')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.show()


# KLD D distance
def create_visibility_graph(time_series):
    graph = nx.DiGraph()
    n = len(time_series)

    for i in range(n):
        for j in range(i + 1, n):
            visibility_condition = all(time_series[k] <= max(time_series[i], time_series[j]) for k in range(i + 1, j))

            if visibility_condition:
                graph.add_edge(i, j)

    # plt.figure()
    # nx.draw(graph, with_labels=True, arrowsize=20, font_size=10, font_color='white', node_color='skyblue')
    # plt.show()
    return graph

def calculate_degree_distribution(graph, direction):
    if direction == 'in':
        degree_sequence = [graph.in_degree(node) + 1 for node in graph]
    elif direction == 'out':
        degree_sequence = [graph.out_degree(node) + 1 for node in graph]
    else:
        raise ValueError("Invalid direction. Use 'in' or 'out'.")
    return degree_sequence

def calculate_kld(pin, pout):
    return stats.entropy(pin, pout)


def kld_series(time_series):
    graph = create_visibility_graph(time_series)
    pin = calculate_degree_distribution(graph, 'in')
    pout = calculate_degree_distribution(graph, 'out')
    kld_value = calculate_kld(pin, pout)
    return kld_value


# print(daily_mean_df['cluster_label'].unique())

kld_list = []
for i in daily_mean_df['cluster_label'].unique():
    df_in_loop = daily_mean_df[(daily_mean_df['cluster_label'] == int(i))]
    df_in_loop = df_in_loop.drop('cluster_label', axis=1)
    kld_value = kld_series(np.array(df_in_loop['Pram']))
    kld_list.append(kld_value)

plt.figure()
plt.plot(kld_list,'-o',color='black')
plt.show()

