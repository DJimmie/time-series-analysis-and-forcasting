""" Perform time series analysis of stock price data"""
# %%
from dependancies_tsa_stocks import *


class TSA():

    def __init__(self,input_parameters):

        self.symbol=input_parameters['symbol']
        self.interval=input_parameters['interval']
        self.feature=input_parameters['feature']


        stock=StockData(self.symbol,interval=self.interval) 

        self.data=stock.get_time_series_data()

        print(self.data.head())
        print(self.data.tail())
        print(self.data.info())



        # self.plot_window(col='close')

        self.univariate_plot(data=self.data,col=self.feature,name=self.symbol)

        self.stationarity_check()

        self.acf()

        self.pacf()

        self.arima()

        


    @staticmethod
    def univariate_plot(data,col,name):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        fig.suptitle(f'Plot Window: {col.upper()}-{name}', fontsize=16, color='yellow')
        ax.set_title(f'{col}')
        ax.set_facecolor('black')
        ax.yaxis.label.set_color('yellow')
        ax.xaxis.label.set_color('yellow')
        ax.set_ylabel(col.upper())
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        ax=data[col].plot(color='#00FFFF',label=f'{col}')

        ax.legend(bbox_to_anchor=(0, 1), loc='lower center', ncol=1)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()

    @staticmethod
    def comparison_plot(data,predicted_data,col,name):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        fig.suptitle(f'Plot Window: {col.upper()}-{name}', fontsize=16, color='yellow')
        ax.set_title(f'{col}')
        ax.set_facecolor('black')
        ax.yaxis.label.set_color('yellow')
        ax.xaxis.label.set_color('yellow')
        ax.set_ylabel(col.upper())
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        ax=data[col].plot(color='#00FFFF',label=f'{col}')
        ax.plot(predicted_data,color='yellow',label=f'predicted {col}')

        ax.legend(bbox_to_anchor=(0, 1), loc='lower center', ncol=1)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()

    def stationarity_check(self):
        """Stationarity check by differencing the data"""

        shift=3
        d = self.data.diff(shift).dropna()

        print(d[self.feature].head())

        print(type(d))

        words=f'{self.feature} with diff({shift})'
        self.univariate_plot(data=d,col='close',name=words)

        self.d=d

    def acf(self):

        # fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        words=f'Autocorrelation\n{self.feature} data'
        plot_acf(self.d[self.feature],title=words)
        plt.show()


    def pacf(self):

        # fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        words=f'Partial Autocorrelation\n{self.feature} data'
        plot_pacf(self.d[self.feature],title=words)
        plt.show()


    def arima(self):

        endog=self.data[self.feature]

        a=ARIMA(endog=endog,order=(1,2,1)).fit(transparams=False)  
        print(a.summary())

        p=a.predict()

        print(p)
        print(type(p))
        

        self.comparison_plot(
            data=self.data,
            predicted_data=p,
            col=self.feature,
            name=self.symbol)



    def prediction(self):
        pass

 

if __name__ == '__main__':

    #-->Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    input_parameters={
        'symbol':'apha',
        'start':None,
        'end':None,
        'feature':'close',
        'interval':'1wk'
        }

    GE=TSA(input_parameters)

    
# %%
