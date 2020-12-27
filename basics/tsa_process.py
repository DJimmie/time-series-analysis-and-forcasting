# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import os

# %%
# ------> get the time series data; setting the Datetime as Index

class RawData():
    """Get the csv file and output a pandas dataframe with the Datetime as an index."""

    def __init__(self,datafile,index_col):
        self.datafile=datafile
        self.index_col=index_col
        self.get_raw_data()

    def get_raw_data(self):

        self.df = pd.read_csv(self.datafile, parse_dates=True, index_col = self.index_col)
        print(self.df.head())
        print(self.df.info())

        # self.p=DataPlot(self.df)

    def data_sub_set(self,start,stop,cols):

        self.subset=self.df.loc[start:stop, cols]

        # self.s=DataPlot(self.subset)

        print(self.subset.head())
        print(type(self.subset))

        return self.subset

    def resample(self,freq):

        self.rs=self.df.resample(freq).mean()

        print(self.rs)

        return self.rs


     
class DataPlot():
    """Visuallizations the various Time Series plots"""

    def __init__(self,df,show=True):
        self.df=df
        self.show=show
        # self.plot1()

    def plot1(self):

        self.df.plot(subplots=True, layout=None, figsize=(10, 8), sharex=True)
        plt.grid()


        if self.show:
            plt.show()

    def shift_plot(self,col,shift=1):

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10, 6))
        fig.suptitle(f'Shift Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')
        ax[0].plot(self.df[col], color='black')
        ax[0].grid()
        shift_name=f'{col}_shift_{shift}'
        shift_value = self.df[col].div(self.df[col].shift(shift))
        ax[1].set_title(f'{col}--->shift({shift})')
        ax[1].plot(shift_value, color='coral')
        ax[1].grid()

        if self.show:
            plt.show()

    def pct_change_plot(self,col,shift=1):

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10, 6))
        fig.suptitle(f'Percent Change Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')

        # ax[0].set_yscale('log')

        ax[0].plot(self.df.loc[:, col], color='black')
        ax[0].grid()

        shift_name=f'{col}_percent_change_{shift}'
        pct = self.df[col].pct_change(shift)*100
        ax[1].set_title(f'{col}--->percent change({shift})')

        # ax[1].set_yscale('log')

        ax[1].plot(pct, color='red')
        ax[1].grid()

        if self.show:
            plt.show()

    def diff_plot(self,col,shift=1):

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10, 6))
        fig.suptitle(f'Difference Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')
        ax[0].plot(self.df.loc[:, col], color='black')
        ax[0].grid()

        shift_name=f'{col}_Differencing_{shift}'
        diff = self.df[col].diff(shift)
        ax[1].set_title(f'{col}--->Difference({shift})')
        # ax[1].set_yscale('log')
        ax[1].plot(diff, color='blue')
        ax[1].grid()

        if self.show:
            plt.show()

    def expanding_window(self,col,shift=1):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6))
        fig.suptitle(f'Expanding Window: {col.upper()}', fontsize=16)
        ax.set_title(f'{col}')
        ax=self.df[col].plot(label=f'{col}')
        ax=self.df[col].expanding().mean().plot(label=f'{col} expanding mean')
        ax=self.df[col].expanding().median().plot(label=f'{col} expanding median')
        ax=self.df[col].expanding().std().plot(label=f'{col} expanding std')

        ax.legend()
        plt.grid()

        if self.show:
            plt.show()
        


# %%
apha=RawData(datafile='APHA.csv',index_col='Date')

# %%
apha_monthly=apha.resample('M')

# %%

apha_monthly

# %%
apha_monthly_plots=DataPlot(df=apha_monthly)
# %%
apha_monthly_plots.plot1()
# %%
apha_plots=DataPlot(df=apha.df)
# %%
apha_plots.plot1()
# %%
apha_weekly=apha.resample('W')
# %%
apha_weekly.head()
# %%
apha_weekly_plots=DataPlot(df=apha_weekly)
# %%
apha_weekly_plots.plot1()
# %%
apha_monthly_plots.shift_plot('Open')
# %%
apha_year_end=apha.resample('A')
# %%
apha_year_end.head()
# %%
apha_year_end_plots=DataPlot(df=apha_year_end)
# %%
apha_year_end_plots.plot1()
# %%
apha.rs
# %%
