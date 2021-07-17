
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

    
    stoploss=-.2
    # how long the model is good for?
    model_timeframe='6h' # how much back data do you need
    
    memory_len=10  # how many time intervals the model looks at
    target_col_index=5 # 
    
    
    def __init__(self,
                 frequency='1min',
                 model_path='lstm10hour_5features/',
                 scaler_path='lstm2_2month_30minfuture_rolling_10_blocks_scalar.pkl'
                ):
        #df = Strategy.format_data(updated_data=start_data,frequency=frequency)
        
        self.model = tensorflow.keras.models.load_model(model_path)
        self.scaler=pickle.load(open(scaler_path,'rb'))
        self.frequency = frequency

        
        
    def predict(self):
        prediction_unscaled = self.model.predict(self.lstm_data)
        prediction=self.scaler.inverse_transform([list(np.append([0]*5,x)) for x in prediction_unscaled])[:,self.target_col_index]

        return({'lastprice':self.df.open_price.iloc[-1]},'prediction30min': prediction[-1]} )
    

    def format_data(self,updated_data,frequency='1min',memory_len = memory_len):
        df = pd.DataFrame(updated_data,columns=['price','datetime'])
        df.datetime =df.datetime
        df.datetime = pd.to_datetime(df.datetime,unit='ms')
        df = df.set_index('datetime')
    
        # Populate Group Indicators
        def nth(listt,k):
            try:
                return listt[k]
            except:
                return()
        
        ######### This is kinda annoying ########
        base=df.index.max()-pd.Timedelta(hours=100000)
        #########################################
        
        #### Make the columns from the data #####
        df = df.groupby(pd.Grouper(freq=frequency,origin=base,closed='right')).agg(
            open_price = pd.NamedAgg(column='price',aggfunc=lambda x: nth(x,0)),
            high=pd.NamedAgg(column='price',aggfunc=max),
            low=pd.NamedAgg(column='price',aggfunc=min),
            close_price = pd.NamedAgg(column='price',aggfunc=lambda x: nth(x,-1)),
            tradecount=pd.NamedAgg(column='price',aggfunc=len),
      #      mean=pd.NamedAgg(column='price',aggfunc=np.mean),
      #      median=pd.NamedAgg(column='price',aggfunc=np.median),
            )
        
        
        # Looks for max of 30 min bins
        min30=[0]*memory_len 
        for i in range(memory_len,df.shape[0]):
            min30.append(max(df.high[i-memory_len:i]))

        df['high30min'] = min30
        
        ###########################################
        
        self.lstm_data = self.lstm_slice(df)
        self.df = df
        self.target_col_index=5  # make sure that the predicted col is noted
        ###########################################

        

    def lstm_slice(self,dataframe,memory_len=memory_len):
        ''' LSTMs dont want every datapoint just the first every 30 min appeneded to the 30 min back'''
        offset = dataframe.shape[0]%memory_len
        
        _data = dataframe[[i%memory_len==offset for i in range(data.shape[0])]].values
        _data = self.scaler.transform(_data)
        predict_on=[]
        for i in range (memory_len, _data.shape[0]):
            predict_on.append(_data[i-memory_len:i]) 
        return (np.array(predict_on))
        
   