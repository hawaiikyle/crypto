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
    
    print('Runner is live and model is loaded')  
    while True:
        
        files = glob.glob('bitcoindata/*.csv')

        if len(files)>0:

            try:
                file = files[0]
        
                print(f'found data{file}')
                
                bitdata= pd.read_csv(file,header=None).values
                
                strategy.format_data(bitdata)
                prediction = pd.DataFrame([strategy.predict()],columns=['lastprice','futureprice30min'])
                prediction['filename'] = file.split('/')[-1].split('.')[0]
                print(['sucessfully predicted',prediction])
                prediction.to_csv('predictiondata/prediction_' +file.split('/')[-1])
    
                os.remove(file)
            except:
                print('found file but ran into errors, likely because of io delay')
                continue
            
            
            
            
            
            
            


if __name__ == '__main__':
    main()