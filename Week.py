import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import powergrid_data as powergrid

data = powergrid.datasets("train.csv", "test_v1.csv")

# NOTE TO SELF: STARTING POINT...(1) Find Week # of the month (2) Print only Week # x (3) Plot Findings (4) Average each day and Plot (5) Plot each week
data_power = data.data[['Date','DateTime','Global_active_power','Global_reactive_power', 'Global_intensity']].dropna()
data2007_may = data_power[(data_power['Date'].dt.month == 5) & (data_power['Date'].dt.year == 2007) & (data_power['Date'].dt.day < 7)]
print(data2007_may)

x = data2007_may['DateTime']
y = data2007_may['Global_active_power']
y2 = data2007_may['Global_reactive_power']
y3 = data2007_may['Global_intensity']

plt.plot(x, y, label='Active Power')
plt.plot(x, y2, label='Reactive Power')
plt.plot(x, y3, label='Global Intensity')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Training Data: Global Active and Reactive Power, and Global Intensity')
plt.show()


#data_07 = data_power[data_power['DateTime'].isin(pd.date_range("05/01/2007", "05/07/2007", freq='min'))]
#mean_2007 = in_2007.mean()
#print(in_2007)
#print(mean_2007)

# NOT QUITE...NaN returned!
# Refer to data['Date] to find 2007; May; and eventually to derive day of the week
#gap_07w1 = data_power['Global_active_power'].where(data_power['DateTime'].isin(pd.date_range("05/01/2007", "05/07/2007", freq='min')))
#print(gap_07w1)

'''
# Full Data Set Global Active and Reactive
x = data_power['DateTime']
y = data_power['Global_active_power']
y2 = data_power['Global_reactive_power']

plt.plot(x, y, label='Active Power')
plt.plot(x, y2, label='Reactive Power')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Training Data: Global Active and Reactive Power')
plt.show()

# Find and Plot the 1st Week of May, 2007
# Identify the first week of May:
#   1st 7 Days? Tues 1, Wed 2, Thurs 3, Fri 4, Sat 5, Sun 6, Mon 7
#   1st Full Week? Mon 7, Tues 8 , Wed 9, Thurs 10, Fri 11, Sat 12, Sun 13

# Use index for first week to access relevant characteristic/trait of data and plot

# Global Active Power

# Global Reactive Power

# Voltage

# Global Intensity

# Sub Metering -- Combined: SM1, SM2, SM3

'''