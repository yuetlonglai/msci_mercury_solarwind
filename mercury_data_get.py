import requests
import pandas as pd
import matplotlib.pyplot as plt

# web crawling to find out when was helios near the mercury orbit
years = ['1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981']
# col_names = ["year", "day", "hour","elapsed_hour","dechr", "min", "sec", "rh", "esh", "clong", "clat", "HGIlong", "br", "bt", "bn", "vp1r", "vp1t", "vp1n", "crot", "np1", "vp1", "Tp1", "vaz", "vel", "Bx", "By", "Bz", "sBx", "sBy", "sBz", "nal", "val"]
col_names = ["year", "day", "hour","dechr", "min", "sec", "rh", "esh", "clong", "clat", "HGIlong", "br", "bt", "bn", "vp1r", "vp1t", "vp1n", "crot", "np1", "vp1", "Tp1", "vaz", "vel", "Bx", "By", "Bz", "sBx", "sBy", "sBz", "nal", "val"]
df_total = pd.DataFrame([])

for year in years:
    url = f'https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/he1_{year}.asc'
    response = requests.get(url)

    if response.status_code == 200:  # Check if the request was successful
        lines = response.text.splitlines()
        # Split each line into values based on spaces (adjust delimiter as needed)
        data = [line.split() for line in lines]

        # Create a DataFrame from the modified data
        df = pd.DataFrame(data, columns=col_names)  # Exclude "year" column

        df_total = pd.concat([df_total, df], ignore_index=True)

df_total = df_total.apply(pd.to_numeric, errors='coerce')
# print(df_total)
extraction_days = df_total[(df_total['rh'] < 0.47) & (df_total['rh'] > 0.31)][['year','day','hour','rh']]
print(extraction_days)


# found out when helios was near mercury, now extract high cadance data
df_all = pd.DataFrame([])
high_cadance_columns = [
    'year', 'day', 'dechr', 'hour', 'min', 'sec', 'rh', 'esh', 'clong', 'clat',
    'HGIlong', 'br', 'bt', 'bn', 'vp1r', 'vp1t', 'vp1n', 'crot', 'np1', 'vp1',
    'Tp1', 'vaz', 'vel', 'Bx', 'By', 'Bz', 'sBx', 'sBy', 'sBz', 'nal', 'val',
    'Tal', 'np2', 'vp2'
]

for i in extraction_days.values:
    if i[1] < 10.0:
        url = f"https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/he1_40sec/H1{int(i[0]-1900)}_00{int(i[1])}.dat"
    elif i[1] < 100.0:
        url = f"https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/he1_40sec/H1{int(i[0]-1900)}_0{int(i[1])}.dat"
    else: 
        url = f"https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/merged/he1_40sec/H1{int(i[0]-1900)}_{int(i[1])}.dat"
    response = requests.get(url)

    if response.status_code == 200:  # Check if the request was successful
        lines = response.text.splitlines()
        # Split each line into values based on spaces (adjust delimiter as needed)
        data = [line.split() for line in lines]

        # Create a DataFrame from the modified data
        df = pd.DataFrame(data, columns=high_cadance_columns)  # Exclude "year" column
        # print(df)
        df = df.drop(0, axis = 0)
        # df = df[(df['hour'].astype(float) == float(i[2]))] # select the correct hours in the day

        df_all = pd.concat([df_all, df], ignore_index=True)
    else:
        # print("Request Unsuccessful : \n" + str(url))
        continue

df_all = df_all.apply(pd.to_numeric, errors = 'coerce')
print(df_all)

# visualise the rh data to check if we got it 
plt.figure()
plt.plot(df_all['rh'],'.')
plt.show()
