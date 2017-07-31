import os
import logging
import argparse, sys, os, logging
import numpy as np
import pandas as pd
import powergrid_data
from hmmlearn import hmm
from itertools import islice
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, DayLocator, AutoDateLocator, DateFormatter, AutoDateFormatter
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

class analyzer:
    hmmModel = {}
    months = {'January': 1, 'Feburary': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    # months = {'January': 1, 'Feburary': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    # 5,10,15, and 20 with threshold 1
    window_size = 5
    window_allow = False

    # 0.5, 1, 1.25 ,1.5, 1.75 and 2 without window
    threshold_size = 0.5

    days = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    def __init__(self, datasets):
        self._train_data, self._test_data = datasets.data, datasets.target

    def time_series(self, time, features):

        self._train_dates = self._train_data[time]
        self._train_features = self._train_data[features]

        self._test_dates = self._test_data[time]
        self._test_features = self._test_data[features]

        self.model_trainer()
        self._not_anomaly = self.model_predictor()

    def updateThreshold(self, threshold = threshold_size):
        self._not_anomaly = self.model_predictor(threshold)

    def parser(self, year, month, day, hour, minutes, seconds):

        parsed_train_dates, parsed_train_features, _ = self._parser_test_train(self._train_dates, self._train_features, year, month, day)
        parsed_test_dates, parsed_test_features, parsed_test_not_anomaly = self._parser_test_train(self._test_dates, self._test_features, year, month, day, hour, minutes, seconds, anomaly = True)

        return parsed_train_dates, parsed_train_features, parsed_test_dates, parsed_test_features, parsed_test_not_anomaly

    def _parser_test_train(self, dates, features, year = 2007, month = None, day = None, hour = None, minutes  = None, seconds = None, anomaly = False):

        features = features[dates.dt.year == year]
        dates = dates[dates.dt.year == year]


        true_value = np.ones(dates.size, dtype=bool)
        # YearFilter = (dates.dt.year == year) if (year != None) else true_value
        MonthFilter = (dates.dt.month == self.months[month]) if (month != None) else true_value
        DayFilter = (dates.dt.day == day) if (day != None) else true_value
        HourFilter = (dates.dt.hour == hour) if (hour != None) else true_value
        MinutesFilter = (dates.dt.minute == minutes) if (minutes != None) else true_value
        SecondsFilter = (dates.dt.seconds == seconds) if (seconds != None) else true_value


        TestFilter = MonthFilter & DayFilter & HourFilter & MinutesFilter & SecondsFilter

        parsed_dates = dates[TestFilter]
        parsed_features = features[TestFilter]

        parsed_dates.reset_index(drop=True, inplace=True)
        parsed_features.reset_index(drop=True, inplace=True)

        parsedNotAnomaly = None

        if anomaly:
            parsedNotAnomaly = np.copy(parsed_features.values)
            parsedNotAnomaly[self._not_anomaly[year][TestFilter]] = None


        return parsed_dates, parsed_features, parsedNotAnomaly

    def outputFile(self):
        totalAnomaly = []
        for year in self._test_dates.dt.year.unique():
            totalAnomaly.extend(self._not_anomaly[year] == False)

        totalAnomaly = np.array(totalAnomaly).astype(int)

        output = pd.DataFrame({'Anomaly Status' : totalAnomaly, 'Probability' : totalAnomaly})
        output.to_csv('pointAnomaly.csv', index=False, header=True)

        return totalAnomaly

    def model_trainer(self):
        self._yearModel()
        self._monthModel()
        self._dayModel()
        # self._dayModel(year = 2007, month = 'March')
        # self._dayModel(year = 2007, month = 'June')

    def model_predictor(self, threshold = threshold_size):
        return self._yearPredictor(threshold)

        # thresholds 0.5, 1, 1.25 ,1.5, 1.75 and 2
    def _yearPredictor(self, threshold, monthPred = True, dayPred = True):
        result = {}
        meanStates = {}
        for date in self._test_dates.dt.year.unique():
            filter = (self._test_dates.dt.year == date)
            feature = self._test_features[filter]
            test_collect_dates = self._test_dates[filter]
            model = self.hmmModel['Year']
            result[date], meanStates[date] = self.feature_predictor(model, feature, threshold)

            if monthPred:
                monthResult = self._monthPredictor(test_collect_dates, feature, dayPred, threshold)

                for month in test_collect_dates.dt.month.unique():
                    monthFilter = (test_collect_dates.dt.month == month)
                    result[date][monthFilter] &= monthResult[month]

        return result


    def _monthPredictor(self, month_test_dates, month_test_features, dayPred, threshold):
        result = {}
        meanStates = {}
        for date in month_test_dates.dt.month.unique():
            filter = (month_test_dates.dt.month == date)
            feature = month_test_features[filter]
            test_collect_dates = month_test_dates[filter]
            model = self.hmmModel['Month']
            result[date], meanStates[date] = self.feature_predictor(model, feature, threshold)

            if dayPred:
                dayResult = self._dayPredictor(test_collect_dates, feature)

                for day in test_collect_dates.dt.day.unique():
                    dayFilter = (test_collect_dates.dt.day == day)
                    result[date][dayFilter] &= dayResult[day]

        return result

    def _dayPredictor(self, day_test_dates, day_test_features, threshold = 1):
        result = {}
        meanStates = {}
        for date in day_test_dates.dt.day.unique():
            filter = (day_test_dates.dt.day == date)
            feature = day_test_features[filter]
            model = self.hmmModel['Day']
            result[date], meanStates[date] = self.feature_predictor(model, feature, threshold)

        return result

    # Ref: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
    def window(seq, n=window_size):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield np.array(result)
        for elem in it:
            result = result[1:] + (elem,)
            yield np.array(result)




    def feature_predictor(self, model, feature, threshold):
        # train_sample, train_state_seq = model.sample(feature.size)
        test_state_seq = model.predict(feature)

        means = []
        variance = []

        for i in range(model.n_components):
            means.append(model.means_[i].flatten()[0])
            variance.append(model.covars_[i].flatten()[0])

        standard_deviation = np.sqrt(variance)

        NotAnomolyBinary = []
        meanStates = []
        test_observation = feature.values

        for i in range(test_observation.size):
            state = test_state_seq[i]
            if (np.abs(means[state] - test_observation[i]) <= threshold):
                NotAnomolyBinary.append(True)
            else:
                NotAnomolyBinary.append(False)
            meanStates.append(means[state])

        # return np.array(NotAnomolyBinary)

        # AnomolyBinary = NotAnomolyBinary == False
        #
        # test_features_anomaly = np.copy(test_observation)
        # test_features_anomaly[NotAnomolyBinary] = None

        return (np.array(NotAnomolyBinary), np.array(meanStates),)


    def _yearModel(self, iter=500, window_accept = window_allow):
        model = None
        if 'Year' in self.hmmModel.keys():
            model = self.hmmModel['Year']
        else:
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", tol = 0.1, n_iter=iter, init_params="st")
        for date in self._train_dates.dt.year.unique():
            filter = (self._train_dates.dt.year == date)
            year_data = self._train_features[filter]

            if window_accept == False:
                 model.fit(year_data)
            else:
                for w in self.window(year_data.values.flatten()):
                    train_window_features = w[:, np.newaxis]
                    model.fit(train_window_features)

        self.hmmModel['Year'] = model

    def _monthModel(self, year = 2007, iter=500, window_accept = window_allow):
        model = None
        if 'Month' in self.hmmModel.keys():
            model = self.hmmModel['Month']
        else:
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", tol = 0.1, n_iter=iter, init_params="st")

        true_value = np.ones(self._train_dates.size, dtype=bool)
        YearFilter = (self._train_dates.dt.year == year) if (year != None) else true_value

        Filter = YearFilter

        month_dates = self._train_dates[Filter]
        month_features = self._train_features[Filter]

        for date in month_dates.dt.month.unique():
            filter = (month_dates.dt.month == date)
            month_data = month_features[filter]
            if window_accept == False:
                model.fit(month_data)
            else:
                for w in self.window(month_data.values.flatten()):
                    train_window_features = w[:, np.newaxis]
                    model.fit(train_window_features)

        self.hmmModel['Month'] = model




    def _dayModel(self, year = 2007, month = 'May', iter=500, window_accept = window_allow):
        model = None
        if 'Day' in self.hmmModel.keys():
            model = self.hmmModel['Day']
        else:
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", tol = 0.1, n_iter=iter, init_params="st")

        true_value = np.ones(self._train_dates.size, dtype=bool)
        YearFilter = (self._train_dates.dt.year == year) if (year != None) else true_value
        MonthFilter = (self._train_dates.dt.month == self.months[month]) if (month != None) else true_value
        # DayFilter = (self._train_dates.dt.day == day) if (day != None) else true_value

        Filter = YearFilter & MonthFilter

        day_dates = self._train_dates[Filter]
        day_features = self._train_features[Filter]

        for date in day_dates.dt.day.unique():
            filter = (day_dates.dt.day == date)
            day_data = day_features[filter]
            if window_accept == False:
                model.fit(day_data)
            else:
                for w in self.window(day_data.values.flatten()):
                    train_window_features = w[:, np.newaxis]
                    model.fit(train_window_features)

        self.hmmModel['Day'] = model