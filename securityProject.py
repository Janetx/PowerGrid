
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
import hmmlearn
from hmmlearn import hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
# from hmmlearn.learn import GaussianHMM
# import hmmlearn as hmm
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sklearn.pipeline
# import hmmlearn


# In[2]:

data = pd.read_csv('train.csv')


# In[3]:

data= data.dropna()


# In[4]:

test_data = pd.read_csv('test_v1.csv')


# In[5]:

test_data= test_data.dropna()


# In[6]:

data


# In[7]:

X_train=data[[ 'Global_active_power', 'Global_reactive_power']]


# In[8]:

X_test= test_data[[ 'Global_active_power', 'Global_reactive_power']]


# In[9]:

# X_test= X_test.dropna()


# In[10]:

X_train


# In[11]:

# X_train['Global_active_power']=X_train['Global_active_power'].values != "NaN"


# In[12]:

# X_train=X_train.dropna()


# In[13]:

# y_train=data['Time'].values


# In[14]:

def to_timestamp( d ):
    ts=pd.Timestamp(d)
    return ts.timestamp()


# In[15]:

y_train = data['Time'].apply(to_timestamp)


# In[16]:

# hmm_model = sklearn.pipeline.make_pipeline(
#             StandardScaler(),
#             hmm.GaussianHMM(n_components=3, covariance_type="full")
#     )


# In[17]:

# model = hmm.GaussianHMM(n_components=3, covariance_type="full")


# In[18]:

hmm_model= hmm.GaussianHMM(n_components=3, covariance_type="full")


# In[19]:

hmm_model.fit(X_train)


# In[20]:

hidden_states = hmm_model.predict(X_test)


# In[21]:

# print("Transition matrix")
# print(hmm_model.transmat_)
# print()


# In[22]:

hidden_states


# In[23]:

print("Transition matrix")
print(hmm_model.transmat_)
print()


# In[24]:

print("Means and vars of each hidden state")
for i in range(hmm_model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", hmm_model.means_[i])
    print("var = ", np.diag(hmm_model.covars_[i]))
    print()


# In[25]:

len(X_train)


# In[26]:

len(y_train)


# In[27]:

fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
#     ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.plot(X_train, ".", c=colour)
    ax.set_title("{0}th hidden state".format(i))

#     Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()


# In[28]:

fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
#     ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.plot(X_test, ".", c=colour)
    ax.set_title("{0}th hidden state".format(i))

#     Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



