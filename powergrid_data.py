
import os
import logging
import pandas as pd


class datasets:
    def __init__(self, train, test):
        train_dir = os.path.dirname(train)
        test_dir = os.path.dirname(test)

        train_data = pd.read_csv(train)
        
        train_data['DateTime'] = train_data['Date'] + ' ' + train_data['Time']
        train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%Y')
        train_data['Time'] = pd.to_datetime(train_data['Time'], format='%H:%M:%S')
        train_data['DateTime'] = pd.to_datetime(train_data['DateTime'], format='%d/%m/%Y %H:%M:%S')
        train_data.dropna(inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        
		
		
        test_data = pd.read_csv(test)
        
        test_data['DateTime'] = test_data['Date'] + ' ' + test_data['Time']
        test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y')
        test_data['Time'] = pd.to_datetime(test_data['Time'], format='%H:%M:%S')
        test_data['DateTime'] = pd.to_datetime(test_data['DateTime'], format='%d/%m/%Y %H:%M:%S')
        test_data.dropna(inplace=True)
        test_data.reset_index(drop=True, inplace=True)
		
        self._train_data = train_data
        self._test_data = test_data

    @property
    def data(self):
        return self._train_data

    @property
    def target(self):
        return self._test_data
