# %%
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os
import sys
import logging

from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 11, 9



sys.path.insert(0,"C:\\Users\\dowdj\\OneDrive\\Documents\\GitHub\\my-modules-and-libraries\\program_work_dir")  # Temporary. Used to help finish development of modules.
import program_work_dir as pwd

register_matplotlib_converters()
# sns.set_style('darkgrid')

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)

# %%
# Create program working folder and its subfolders
config_parameters={'database server':{'sqlite_server':'N/A'}}
client=pwd.ClientFolder(os.path.basename(__file__),config_parameters)
ini_file=f'c:/my_python_programs/{client}/{client}.ini'

log_file=f'c:/my_python_programs/{client}/{client}_log.log'
logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format=' %(asctime)s -%(levelname)s - %(message)s')
logging.info('Start')


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

        subset=pd.DataFrame(self.subset)
        print(subset.head())
        return subset

    def resample(self,freq):

        self.rs=self.df.resample(freq).mean()

        print(self.rs)

        return self.rs


     
class DataPlot():
    """Visuallizations the various Time Series plots"""

    def __init__(self,df,show=False):
        self.df=df
        self.show=show
        # self.plot1()

    def plot1(self):

        self.df.plot(subplots=True, layout=None, figsize=(15, 10), sharex=True)
        plt.grid()


        if self.show:
            plt.show()

    def shift_plot(self,col,shift=1,yscale='linear'):

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10, 6))
        fig.suptitle(f'Shift Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')
        ax[0].set_yscale(yscale)
        ax[0].plot(self.df[col], color='black')
        ax[0].grid()
        shift_name=f'{col}_shift_{shift}'
        shift_value = self.df[col].div(self.df[col].shift(shift))
        ax[1].set_title(f'{col}--->shift({shift})')
        ax[1].set_yscale(yscale)
        ax[1].plot(shift_value, color='coral')
        ax[1].grid()

        if self.show:
            plt.show()

    def pct_change_plot(self,col,shift=1,yscale='linear'):

        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10, 6))
        fig.suptitle(f'Percent Change Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')

        ax[0].set_yscale(yscale)

        ax[0].plot(self.df.loc[:, col], color='black')
        ax[0].grid()

        shift_name=f'{col}_percent_change_{shift}'
        pct = self.df[col].pct_change(shift)*100
        ax[1].set_title(f'{col}--->percent change({shift})')

        ax[1].set_yscale(yscale)
        ax[1].bar(pct.index,pct, color='red')
        # ax[1].plot(pct, color='red')
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
        # ax=self.df[col].expanding().sum().plot(label=f'{col} expanding sum')

        ax.legend()
        plt.grid()

        if self.show:
            plt.show()
        
    def rolling_window(self,col,period=1):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6))
        fig.suptitle(f'Rolling Window: {col.upper()}', fontsize=16)
        ax.set_title(f'{col}')
        ax=self.df[col].plot(label=f'{col}')
        mu=self.df[col].rolling(period,center=False).mean()
        ax=self.df[col].rolling(period,center=False).mean().plot(label=f'{col} {period} Moving Avg')
        sigma=self.df[col].rolling(period,center=False).std()
        minus_2sigma=(mu-2*sigma)
        plus_2sigma=(mu+2*sigma)
        ax=minus_2sigma.plot(label=f'{col} {period} -2 sigma',color='red',alpha=0.2)
        ax=plus_2sigma.plot(label=f'{col} {period} +2 sigma',color='red',alpha=0.2)
        ax=plt.fill_between(self.df[col].index,plus_2sigma,minus_2sigma,alpha=0.2)
        print(f'sigma:{minus_2sigma}')
        plt.legend()
        plt.grid()
        
        if self.show:
            plt.show()

        self.macd('Adj Close')

    def hist_plot(self,col):
        fig, ax = plt.subplots(nrows=2,ncols=1,sharex=False,figsize=(10, 6))
        fig.suptitle(f'Histogram Plot: {col.upper()}', fontsize=16)
        ax[0].set_title(f'{col}')
        ax[0].plot(self.df.loc[:, col], color='black')
        ax[0].grid()

        ax[1].set_title(f'{col}--->Histogram')
        # ax[1].set_yscale('log')
        ax[1].hist(self.df[col],bins=100, density=True, color='blue')
        ax[1].grid()


    def macd(self,col):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6))
        fig.suptitle(f'MACD: {col.upper()}', fontsize=16)
        ax.set_title(f'{col}')

        exp1 = self.df[col].ewm(span=12, adjust=False).mean()
        exp2 = self.df[col].ewm(span=26, adjust=False).mean()
        macd = exp1-exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()

        ax=plt.plot(macd.index,macd,label='AMD MACD', color = '#4C0099')
        ax=plt.plot(exp3.index,exp3,label='Signal Line', color='#FF9933')
        plt.legend()
        plt.grid()





class TSAnalysis():
    """TS Analysis"""

    def __init__(self,df):
        self.df=df

    def decompose (self,col):
        # decomposition = sm.tsa.seasonal_decompose(pd.DataFrame(self.df[col]), model='Additive')
        # fig = decomposition.plot()
        a=self.df[col]
        b=a.index
        c=pd.infer_freq(b)
        
        d = pd.Series(a, index=pd.date_range(b.date[0], b.date[len(a)-1], periods=None, freq=c), name = col)
        
        print(d)
        # x=pd.Series(p.values,index=p.index.values)
        stl = STL(d, seasonal=7)
        res = stl.fit()
        fig = res.plot()
        plt.show()


# %%

def arima(data):
    model = ARIMA(data['Close'], order=(1,1,1))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data)+5)
    print(yhat)

## FUNCTIONS----------------------------------FUNCTIONS----------------------------------FUNCTIONS
def get_config_values(section,option):
    """Used to retrieve values from the program's configuration file."""
    config=pwd.configparser.ConfigParser()
    config.read(ini_file)

    return config[section][option]


def exit_operations():
    """Operations to perform prior to the program exit"""
    # remove images from the image folder after each session

    # save the table as a json file in the working directory
    pass


## MAIN----------------------------------MAIN----------------------------------MAIN
# %%

data="TSLA.csv"    #"CBWTF.csv" #"KSHB.csv" #APHA.csv" #"quicken_dsv.csv"

data_file=f'c:/my_python_programs/{client}/{data}'



# %%
# bank=RawData(datafile=data_file,index_col='postedOn')
# %%
apha=RawData(datafile=data_file,index_col='Date')
# %%
apha_plots=DataPlot(df=apha.df)

arima(apha.df)
print(apha.df.tail())


apha_2020=apha.data_sub_set('2020','2020',cols='Adj Close')

plt.show()
# %%
apha_2020_plots=DataPlot(df=apha_2020)
# %%
apha_2020_plots.rolling_window('Adj Close',20)
# %%

plt.show()
