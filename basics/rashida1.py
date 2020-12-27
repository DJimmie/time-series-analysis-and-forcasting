"""Time Series Data Visualization in Python. Based on blog by rashida048."""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar


# %%
df = pd.read_csv("APHA.csv", parse_dates=True, index_col = "Date")
df.head()

print(df.head())
# %%
df['Volume'].plot(figsize=(8, 6))
# %%
df.plot(subplots=True, figsize=(10,12))
# %%
df_month = df.resample("M").mean()

# %%
df_month

# %%
fig, ax = plt.subplots(figsize=(10, 6))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.bar(df_month['2020':].index,
df_month.loc['2020':, "Open"],
width=25, align='center')


# %%
df_month['Volume'].plot(figsize=(8, 6))
# %%
df_week = df.resample("W").mean()
# %%
start, end = '2015-01', '2020-12'
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.loc[start:end, 'Volume'], marker='.', linestyle='-', linewidth = 0.5, label='Daily', color='black')
ax.plot(df_week.loc[start:end, 'Volume'], marker='o', markersize=4, linestyle='-', label='Weekly', color='coral')
label='Monthly'
color='violet'
ax.set_ylabel("Open")
ax.legend()
# %%

df_month['2015']['Open'].plot()






# %%

df_week.loc['2020':, "Open"].plot()


# %%
df_month.loc['2020':, "Open"].plot()
# %%
df_7d_rolling = df.rolling(7, center=True).mean()
start, end = '2016-06', '2017-05'
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Volume'], marker='.', linestyle='-', 
        linewidth=0.5, label='Daily')
ax.plot(df_week.loc[start:end, 'Volume'], marker='o', markersize=5, 
        linestyle='-', label = 'Weekly mean volume')
ax.plot(df_7d_rolling.loc[start:end, 'Volume'], marker='.', linestyle='-', label='7d Rolling Average')
ax.set_ylabel('Stock Volume')
ax.legend()
# %%
df_month_rolling=df_month.rolling(3,center=False).mean()
start, end = '2020-06', '2020-11'
fig, ax = plt.subplots()
ax.plot(df_month.loc[start:end, 'Open'], marker='.', linestyle='-', 
        linewidth=0.5, label='Monthly')
ax.plot(df_month_rolling.loc[start:end, 'Open'], marker='o', markersize=5, 
        linestyle='-', label = 'Moving Avg')

ax.set_ylabel('Stock Volume')
ax.legend()
# %%
df['Change'] = df.Close.div(df.Close.shift())
df['Change'].plot(figsize=(20, 8), fontsize = 16)
# %%
df.loc['2019':'2020','Change'].plot(subplots=True,figsize=(20, 8), fontsize = 16)

# %%
df.plot(subplots=True, figsize=(10,12))
# %%
df['Change'] = df.div(df.shift())
df['Change'].plot(subplots=True,figsize=(20, 8), fontsize = 16)

# %%
df_month.loc[:, 'pct_change'] = df.Close.pct_change()*100
fig, ax = plt.subplots()
df_month['pct_change' ].plot(kind='bar', color='coral', ax=ax)
# ax.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
ax.legend()
# %%
df.High.diff().plot(figsize=(10, 6))
# %%
df.High.diff()
# %%
fig, ax = plt.subplots()
ax = df.High.plot(label='High')
ax = df.High.expanding().mean().plot(label='High expanding mean')
ax = df.High.expanding().std().plot(label='High expanding std')
ax.legend()
# %%
all_month_year_df = pd.pivot_table(df_month, values="Open",
                                   index=df_month.index,
                                   columns=['Close'],
                                   fill_value=0,
                                   margins=True)
named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]] # name months
all_month_year_df = all_month_year_df.set_index(named_index)
all_month_year_df
# %%
calendar.month(2020,12)
# %%
calendar.monthcalendar(2020,12)
# %%
calendar.monthrange(2020,12)
# %%
# %%
