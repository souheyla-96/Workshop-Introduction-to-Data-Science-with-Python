#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np

# Replace with the actual dataset path from Padoc
df = pd.read_csv("C:/Users/FAROUK/Downloads/AAPL.csv")

# Inspect the first few rows
print(df.head())


# In[4]:


df[['Date', 'Open']]


# In[4]:


df[['Date', 'Open']]


# In[122]:


print(df.columns)  #same with print


# In[44]:


df.sample(3)   # 3 lignes aléatoires


# In[10]:


df.tail() # last 5 lines


# In[13]:


df.dtypes             # types of each column


# In[14]:


df.info() # full summary


# In[15]:


df.describe()                       # numerical statistical summary


# In[16]:


df.describe().round(2)   #Round all numerical statistics in the summary to 2 decimal places for better readability


# In[20]:


df['Open'].value_counts()       # frequency of values


# In[21]:


pd.set_option('display.max_columns',None) # display all columns\n
print(df)


# In[23]:


pd.set_option('display.precision',2) # number of decimal places
print(df)


# In[25]:


df ['Date'] # single column


# In[26]:


df [['Date' , 'Volume']] # multiple columns


# In[27]:


df.iloc[0] # first line


# In[28]:


df.iloc[0:5] # lines 0 to 4\n


# In[41]:


df.loc [df['Volume'] >100000000] # conditional filtering


# In[46]:


df ['mean'] = (df['High'] + df['Low']) / 2


# In[48]:


pd.reset_option('display.max_columns') # reset display option in default mode\n
print(df)


# In[52]:


df.drop ('mean' , axis= 1 , inplace = True) 
        # removes the column named 'mean' from the DataFrame df by specifying \n,
        # axis=1 to indicate a column operation, and inplace=True to apply the change \n,
        # directly without reassigning the DataFrame.\n,
print(df)


# In[57]:


df[df['Volume'] == df['Volume'].min()] # Display the row  with the minimum volume


# In[58]:


df.sort_values (by = 'Volume' , ascending = False) #sorting values


# In[60]:


#MATPLOTLIB

import matplotlib.pyplot as plt

#SEABORN

import seaborn as sns


# In[88]:


sns.histplot(df['Volume'], bins=40)
plt.title("Distribution du Volume échangé")


# In[87]:


sns.scatterplot(x='Date', y='Volume', hue='Low', data=df)
plt.title("Volume échangé en fonction du temps, coloré par le prix bas (Low)")


# In[90]:


# Heatmap of correlation between numeric variables,
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# In[123]:


import plotly.express as px
# Use existing numeric and categorical columns
px.scatter(df, x='Date', y='Adj Close', color='Volume', title="Adj Close vs Date Colored by Volume")


# In[95]:


# Importing the pyplot interface from Matplotlib
import matplotlib.pyplot as plt


# In[106]:


plt.plot(df['Date'], df['Volume'], color="pink")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Évolution du Volume échangé")
plt.show()


# In[107]:


# You can customize the line style, marker and color
plt.plot(df['Date'], df['Volume'], linestyle='--', marker='v', color='r')
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Évolution du Volume échangé")
plt.show()


# In[110]:


df.Date = pd.to_datetime(df.Date)
df.Date.dtype


# In[114]:


# Prix de clôture
plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Close)
plt.ylabel("Prix de cloture")
plt.title("Evolution du prix de cloture de APPLE")
plt.show()


# In[115]:


plt.figure(figsize=(12, 6))
plt.plot(df.Date, df.Low, label="Bas")
plt.plot(df.Date, df.High, label="Haut")
plt.ylabel("Prix")
plt.legend()
plt.show()


# In[116]:


# Point Clouds (scatter plot)
plt.figure(figsize=(10,8))
plt.scatter(df.Volume, df.Close)
plt.xlabel('Volume négocié')
plt.ylabel("Prix de cloture")
plt.show()


# In[117]:


import numpy as np

import plotly

import plotly.graph_objects as go

import plotly.offline as pyo

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

fig = go.Figure(data=go.Scatter(x=df['Date'], y=df['Volume'], mode='markers'))

fig.show()


# In[121]:


fig = go.Figure(data=[go.Histogram(x=df['Close'])])
fig.show()


# In[ ]:




