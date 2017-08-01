import os
import logging
import argparse, sys, os, logging
import numpy as np
import pandas as pd
import powergrid_data
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, DayLocator, AutoDateLocator, DateFormatter, AutoDateFormatter
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

class analyzer:

    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    def __init__(self, datasets):
        self._train_data, self._test_data = datasets.data, datasets.target

    def time_series(self, time, features):

        self._train_dates = self._train_data[time]
        self._train_features = self._train_data[features]

        self._test_dates = self._test_data[time]
        self._test_features = self._test_data[features]

    def parser(self, year, month, day, hour, minutes, seconds):

        parsed_train_dates, parsed_train_features = self._parser_test_train(self._train_dates, self._train_features, year, month, day)
        parsed_test_dates, parsed_test_features = self._parser_test_train(self._test_dates, self._test_features, year, month, day, hour, minutes, seconds)

        return parsed_train_dates, parsed_train_features, parsed_test_dates, parsed_test_features

    def _parser_test_train(self, dates, features, year = None, month = [None], day = None, hour = None, minutes  = None, seconds = None):
        true_value = np.ones(dates.size, dtype=bool)
        YearFilter = (dates.dt.year == year) if (year != None) else true_value
        MonthFilter = true_value == False
        for m in month:
            MonthFilter |= (dates.dt.month == self.months[m]) if (m != None) else true_value
            
            
        DayFilter = (dates.dt.day == day) if (day != None) else true_value
        HourFilter = (dates.dt.hour == hour) if (hour != None) else true_value
        MinutesFilter = (dates.dt.minute == minutes) if (minutes != None) else true_value
        SecondsFilter = (dates.dt.seconds == seconds) if (seconds != None) else true_value


        TestFilter = YearFilter & MonthFilter & DayFilter & HourFilter & MinutesFilter & SecondsFilter

        parsed_dates = dates[TestFilter]
        parsed_features = features[TestFilter]

        parsed_dates.reset_index(drop=True, inplace=True)
        parsed_features.reset_index(drop=True, inplace=True)

        return parsed_dates, parsed_features
