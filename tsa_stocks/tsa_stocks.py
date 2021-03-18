""" Perform time series analysis of stock price data"""
# %%
from dependancies_tsa_stocks import *


class TSA():

    def __init__(self,input_parameters):

        self.data_source=input_parameters['source']
        self.symbol=input_parameters['symbol']
        self.interval=input_parameters['interval']
        self.feature=input_parameters['feature']
        self.end=input_parameters['end']
        self.start=input_parameters['start']
        self.period=input_parameters['period']

        if self.data_source=='stocks':
            self.tsa_on_stocks()
        elif self.data_source=='random_walk':
            self.tsa_on_random_walk()

    def tsa_on_stocks(self):
        stock=StockData(
            self.symbol,
            interval=self.interval,
            start_date=self.start,
            end_date=self.end,
            period=self.period)
        self.data=stock.get_time_series_data()
        self.univariate_plot(data=self.data,col=self.feature,name=self.symbol)

    def tsa_on_random_walk(self):

        # Retrieving data from the random_walk_sim
        data_dict=random_walk_sim(self.feature)

    #     data_dict={
    #     'train_size':train_size,
    #     'train':train,
    #     'test':test,
    #     'df':df
    # }

        self.data=data_dict['df']   ##--->model training data

        print(self.data.tail())

        # Plotting the training data time series
        self.univariate_plot(
            data=self.data,
            col=self.feature,
            name='Random Walk Data')

        # call the differencing function
        self.stationarity_check(data_dict['df'])

        # Get the ACF & PACF data & plots
        self.acf()
        self.pacf()

        # Run ARIMA
        model=self.arima()

        print(model.summary())

        # get predictions on test data
        test_data=data_dict['test']
        prediction=model.predict(
            start=len(self.data),
            end=len(self.data)+len(test_data),
            typ='levels')
        print(prediction)

        print(f'forecast\n{model.forecast()[0]}')

        self.comparison_plot(
            data=self.data,
            predicted_data=prediction,
            test_data=test_data,
            col=self.feature,
            name=' ')

        train=data_dict['train']
        history=[x for x in train]
        predictions = list()
        for t in range(len(test_data)):
            model = ARIMA(history, order=(1,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_data[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))


        self.comparison_plot(
            data=self.data,
            predicted_data=predictions,
            test_data=test_data,
            col=self.feature,
            name=' ')

        self.calculate_rmse(test_data,predictions)


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
    def comparison_plot(data,predicted_data,test_data,col,name):
        fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        fig.suptitle(f'Plot Window: {col.upper()}-{name}', fontsize=16, color='yellow')
        ax.set_title(f'{col}')
        ax.set_facecolor('black')
        ax.yaxis.label.set_color('yellow')
        ax.xaxis.label.set_color('yellow')
        ax.set_ylabel(col.upper())
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Training data plot
        ax=data[col].plot(color='#00FFFF',label=f'{col}')

        # predicted data plot. Can be ither a list or dataframe
        test_data_x=[len(data)+i for i in range(0,len(test_data))]
        if type(predicted_data)==list:
            ax.plot(test_data_x,predicted_data,color='yellow',label=f'predicted {col}')
        else:
            ax.plot(predicted_data,color='yellow',label=f'predicted {col}')
        
        # Test data plot
        ax.plot(test_data_x,test_data,color='#7CFC00',label=f'test {col}')

        ax.legend(bbox_to_anchor=(0, 1), loc='lower center', ncol=1)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()


    @staticmethod
    def calculate_rmse(test_data, predictions):
        # evaluate forecasts
        rmse = sqrt(mean_squared_error(test_data, predictions))
        print('Test RMSE: %.3f' % rmse)
        # plot forecasts against actual outcomes
        plt.plot(test_data)
        plt.plot(predictions, color='red')
        plt.show()

    def stationarity_check(self,df):
        """Stationarity check by differencing the data"""

        shift=2
        d = df.diff(shift).dropna()

        print(d[self.feature].head())

        print(type(d))

        words=f'{self.feature} with diff({shift})'
        self.univariate_plot(data=d,col=self.feature,name=words)

        self.d=d[self.feature]

        # print(f'dddddd---->\n{self.d}')

    def acf(self):

        # fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        words=f'Autocorrelation\n{self.feature} data'
        plot_acf(self.d,title=words)
        plt.show()


    def pacf(self):

        # fig, ax = plt.subplots(nrows=1,ncols=1,sharex=False,figsize=(10, 6),facecolor='black')
        words=f'Partial Autocorrelation\n{self.feature} data'
        plot_pacf(self.d,title=words)
        plt.show()


    def arima(self):

        endog=self.data[self.feature]
        # endog=self.d
        a=ARIMA(endog=endog,order=(1,0,1)).fit()  
        
        return a


def random_walk_sim(feature):

    PERIODS=360
    # generate random walk
    seed(42)
    random_walk=list()
    random_walk.append(-1 if random()<0.5 else 1)
    for i in range(1,PERIODS):
        movement=-1 if random()<0.5 else 1
        value=random_walk[i-1]+movement
        random_walk.append(value)

    # Setting up the training & test data
    train_size=int(len(random_walk)*.80)
    train, test=random_walk[0:train_size], random_walk[train_size:]

    # run the Dickey-Fuller test for stationarity
    ADF(train)

    # creating a dataframe
    idx = pd.date_range("2018-01-01", periods=train_size, freq="M")
    df=pd.DataFrame(idx, columns=['date'],)
    df[feature]=train

   #Generating dictionary to contain function outputs
    data_dict={
        'train_size':train_size,
        'train':train,
        'test':test,
        'df':df
    }

    return data_dict

def ADF(data):
    """Augmented Dickey-Fuller test"""

    result=adfuller(data)

    print(type(result))
    print(type(result[4]))

    print(f'ADF Statistic:{result[0]}')
    print(f'p-value:{result[1]}')
    print('Critical Values:\n')

    for k,v in result[4].items():
        print(f'{k}--->,{v}')
        
    
if __name__ == '__main__':

    #-->Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # input_parameters={
    #     'source':'random_walk',
    #     'symbol':'cron',
    #     'start':None,
    #     'end':None,
    #     'feature':'data',
    #     'interval':'1d',
    #     'period':'2y'
    #     }


    input_parameters={
        'source':'stocks',
        'symbol':'cron',
        'start':None,
        'end':None,
        'feature':'close',
        'interval':'1d',
        'period':'2y'
        }

    GE=TSA(input_parameters)

    
# %%
