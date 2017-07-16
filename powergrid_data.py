
import os
import logging
import pandas as pd


class datasets:
    def __init__(self, train, test):
        train_dir = os.path.dirname(train)
        test_dir = os.path.dirname(test)

        hdf_file = os.path.join("data", "hdf", "hdf_data.h5")

        if not (os.path.isfile(hdf_file)):
            logging.debug("Creating HDF5 Training and Testing Files")
            train_data = pd.read_csv(train, parse_dates=True)
            test_data = pd.read_csv(test, parse_dates=True)

            train_data.to_hdf(hdf_file, "train_data", mode='w', format='table')
            test_data.to_hdf(hdf_file, 'test_data', mode='a', format='table')

        self._train_data = pd.read_hdf(hdf_file, "train_data")
        self._test_data = pd.read_hdf(hdf_file, 'test_data')

    @property
    def data(self):
        return self._train_data

    @property
    def target(self):
        return self._test_data
