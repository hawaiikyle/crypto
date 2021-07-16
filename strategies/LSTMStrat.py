
from pandas import DataFrame
import numpy as np
import pandas as pd
#----- Here are the functions For the Strategy ---------

def crossed(series1, series2, direction=None):
    
    # Just tells me when a trend crosses another trend 
    
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)

    if isinstance(series2, (float, int, np.ndarray, np.integer, np.floating)):
        series2 = pd.Series(index=series1.index, data=series2)

    if direction is None or direction == "above":
        above = pd.Series((series1 > series2) & (
            series1.shift(1) <= series2.shift(1)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (
            series1.shift(1) >= series2.shift(1)))

    if direction is None:
        return above or below

    return above if direction == "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")




# ------ Strategy -----------


# Note date is bins of say... 15 min or so. 

class Strategy:
    """
    idea:
        buys and sells on crossovers - doesn't really perfom that well 
    
    """
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0.01
    }
    # Number of time bins for short and long term
    short_term = 8
    long_term = 21
    
    stoploss=-.2
    # how long the model is good for?
    model_timeframe='4h'
    memory_len=10
    target_col_index=5
    
    def __init__(self, start_data=[],frequency='10min'):
        df = Strategy.format_data(updated_data=start_data,frequency=frequency)
        

        self.df =df

        self.frequency = frequency

        
    @staticmethod
    def format_data(updated_data,frequency,memory_len = memory_len  ):
        df = pd.DataFrame(updated_data,columns=['price','datetime'])
        df.datetime =df.datetime
        df.datetime = pd.to_datetime(df.datetime)
        df = df.set_index('datetime')
    
        # Populate Group Indicators
        def nth(listt,k):
            try:
                return listt[k]
            except:
                return()
    
        df = df.groupby(pd.Grouper(freq='5min',origin=base,closed='right')).agg(
            open_price = pd.NamedAgg(column='price',aggfunc=lambda x: nth(x,0)),
            high=pd.NamedAgg(column='price',aggfunc=max),
            low=pd.NamedAgg(column='price',aggfunc=min),
            close_price = pd.NamedAgg(column='price',aggfunc=lambda x: nth(x,-1)),
            tradecount=pd.NamedAgg(column='price',aggfunc=len),
     #       mean=pd.NamedAgg(column='price',aggfunc=np.mean),
      #      median=pd.NamedAgg(column='price',aggfunc=np.median),
            
    
            )
        min30=[0]*memory_len 
        for i in range(memory_len,df.shape[0]):
            min30.append(max(df.high[i-memory_len:i]))
        df['high30min'] = min30
        return df[memory_len+1:]

    @staticmethod
    def populate_indicators( dataframe: DataFrame) -> DataFrame:
    # this will populate all the indicators that will determine the buy/sell flags
        Y_pred_unscaled=model.predict(test)
        Y_pred=np.append([0]*memory_len,scaler.inverse_transform(
                [list(np.append([0]*target_col_index,x)) for x in Y_pred_unscaled])[:,5])
        
        dataframe['lstm_30_min_high'] = Y_pred
        

        return dataframe[memory_len+1:]
    
    @staticmethod 
    def populate_buy_trend(dataframe: DataFrame):
        """
    
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                cdataframe['close_price']< dataframe['lstm_30_min_high'])
            ),
            'buy'] = 1
    
        return dataframe
    
    @staticmethod
    def populate_sell_trend( dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                dataframe['close_price']<dataframe['lstm_30_min_high']
            ),
            'sell'] = 1
        return dataframe
    
# Helper functions 
    def strategy_status(self):
        if self.df.buy.iloc[-1] ==1:
            return ('Buy')
        if self.df.sell.iloc[-1]==1:
            return('Sell')
        return('idk')
        
        