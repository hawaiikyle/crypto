#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 19:45:05 2021

@author: kyleevans-lee
"""
import glob
import os
import pandas as pd
from strategies.LSTMStrat import Strategy


def main():
    
    strategy = Strategy(model_path='models/lstm10hour_5features/',
                        scaler_path='models/lstm10_hour_5features_scaler.pkl')
    
    
    while True:
        
        files = glob.glob('bitcoindata/*.csv')
        if len(files)>0:
            print('found data')
            file = files[0]
            bitdata= pd.read_csv(file,header=None).values
            
            strategy.format_data(bitdata)
            prediction = pd.DataFrame([strategy.predict()],columns=['lastprice','futureprice30min'])
            print(prediction)
            print(glob.glob('bitcoindata/*.csv'))
            prediction.to_csv('predictiondata/predictiondata.csv')
            try:
                os.remove(file)
            except:
                continue
            
            
            
            
            
            
            


if __name__ == '__main__':
    main()