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
    
    strategy = Strategy()
    
    
    while True:
        
        files = glob.glob('bitcoindata/*.csv')
        if len(files)>0:
            print('found data')
            file = files[0]
            bitdata= pd.read_csv(file).values
            
            strategy.format_data(bitdata)
            prediction = pd.DataFrame(strategy.predict())
            print(prediction)
            
            prediction.to_csv('predictiondata/predictiondata.csv')
            os.rm(file)
            
            
            
            
            
            
            
            
            
            
            
            os.remove(file)


if __name__ == '__main__':
    main()