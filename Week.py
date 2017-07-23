import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import powergrid_data as powergrid

data = powergrid.datasets("train.csv", "test_v1.csv")

# NOTE TO SELF: STARTING POINT...1) Plot each week of the Month of May on a single Plot (2) More Generalized Line
data_power = data.data[['Date','DateTime','Global_active_power','Global_reactive_power', 'Global_intensity']].dropna()
data2007_may = data_power[(data_power['Date'].dt.month == 5) & (data_power['Date'].dt.year == 2007) & (data_power['Date'].dt.day < 7)]

'''
#data_power = data.data[['Date','DateTime','Global_active_power']].dropna()
#data2007_may = data_power[(data_power['Date'].dt.month == 5) & (data_power['Date'].dt.year == 2007)]
'''

#Normalization
df = pd.DataFrame(data2007_may)
'''
avg_gactive = df['Global_active_power'].mean()
normalized_data2007_may = data2007_may['Global_active_power']/avg_gactive
'''
# (1) Take Average of Full Data Set
avg_gactive = df['Global_active_power'].mean()
avg_greactive = df['Global_reactive_power'].mean()
avg_gintensity = df['Global_intensity'].mean()
# (2) Divide each data point against the average = (Data/Avg)
normalized_data2007_may_ga = data2007_may['Global_active_power']/avg_gactive
normalized_data2007_may_gr = data2007_may['Global_reactive_power']/avg_gactive
normalized_data2007_may_gi = data2007_may['Global_intensity']/avg_gactive

# Plot
x = data2007_may['DateTime']
y = data2007_may['Global_active_power']
yn = normalized_data2007_may_ga

y2 = data2007_may['Global_reactive_power']
y2n = normalized_data2007_may_gr

y3 = data2007_may['Global_intensity']
y3n = normalized_data2007_may_gi

plt.plot(x, y, label='Active Power')
plt.plot(x, yn, label='Normalized Active Power')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Training Data: Global Active and Normalized Global Active Power')
plt.legend()
plt.show()

plt.plot(x, y2, label='Reactive Power')
plt.plot(x, y2n, label='Normalized Reactive Power')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Training Data: Global Reactive and Normalized Global Reactive Power')
plt.legend()
plt.show()

plt.plot(x, y3, label='Global Intensity')
plt.plot(x, y3n, label='Normalized Global Intensity')
plt.xlabel('DateTime')
plt.ylabel('Intensity')
plt.title('Training Data: Global Intensity and Normalized Global Intensity')
plt.legend()
plt.show()




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