#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

# Replace with the actual dataset path from Padoc
df = pd.read_csv("C:/Users/FAROUK/Downloads/archive/spot.csv")

# Inspect the first few rows
print(df.head())


# In[4]:


df[['track_id', 'track_name']]


# In[5]:


print(df.columns)  #same with print


# In[6]:


df.sample(3)   # 3 lignes aléatoires


# In[7]:


df.tail() # last 5 lines


# In[8]:


df.dtypes   


# In[9]:


df.info() # full summary


# In[10]:


df.describe()                       # numerical statistical summary


# In[11]:


df.describe().round(2)   #Round all numerical statistics in the summary to 2 decimal places for better readability


# In[13]:


df['track_number'].value_counts()       # frequency of values


# In[14]:


pd.set_option('display.max_columns',None) # display all columns\n
print(df)


# In[15]:


pd.set_option('display.precision',2) # number of decimal places
print(df)


# In[16]:


df ['album_type'] # single column


# In[17]:


df [['track_name' , 'album_type']] # multiple columns


# In[18]:


df.iloc[0] # first line


# In[19]:


df.iloc[0:5] # lines 0 to 4\n


# In[24]:


df.loc [df['track_duration_ms'] >220000] # conditional filtering


# In[40]:


df ['totale album duration'] = (df['track_duration_ms']) * (df['track_number']) 


# In[41]:


pd.reset_option('display.max_columns') # reset display option in default model
print(df)


# In[31]:


df.drop ('totale album duration' , axis= 1 , inplace = True) 
        # removes the column named 'mean' from the DataFrame df by specifying \n,
        # axis=1 to indicate a column operation, and inplace=True to apply the change \n,
        # directly without reassigning the DataFrame.\n,
print(df)


# In[32]:


df[df['album_total_tracks'] == df['album_total_tracks'].min()] # Display the row  with the minimum volume


# In[33]:


df.sort_values (by = 'album_total_tracks' , ascending = False) #sorting values


# In[34]:


#Rename 'Unnamed: 0' to 'index'
df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
# Display the first rows to confirm
df.head() # first 5 lines


# In[36]:


nb = df.isnull().sum()
print(nb)
df [df.isnull().any (axis = 1)]


# In[37]:


#MATPLOTLIB

import matplotlib.pyplot as plt

#SEABORN

import seaborn as sns


# In[50]:


sns.histplot(df['track_popularity'], bins=40)
plt.title("Distribution of track popularity")


# In[54]:


sns.scatterplot(x='artist_popularity', y='track_popularity', hue='track_duration_ms', data=df)
plt.title("track popularity en artist popularity, coloré par le track_duration_ms")


# In[47]:


# Heatmap of correlation between numeric variables,
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# In[56]:


import plotly.express as px
# Use existing numeric and categorical columns
px.scatter(df, x='artist_popularity', y='artist_followers', color='track_popularity', title="artist followers vs artist_popularity Colored by track popularity")


# In[57]:


# Importing the pyplot interface from Matplotlib
import matplotlib.pyplot as plt


# In[75]:


year=[ 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


# In[73]:


number_users=[ 4562, 2783, 7685, 2346, 9643, 8956, 7643, 3454, 4567,3456 , 2345, 12345]


# In[76]:


plt.plot(year, number_users, color="green")
plt.xlabel("year")
plt.ylabel("number users")
plt.title("Evolution of number users")
plt.show()


# In[77]:


# You can customize the line style, marker and color
plt.plot(year, number_users, linestyle='--', marker='v', color='r')
plt.xlabel("year")
plt.ylabel("number users")
plt.title("Evolution of number users")
plt.show()


# In[79]:


plt.figure(figsize=(12, 6))
plt.plot(year, number_users, color="green")
plt.xlabel("year")
plt.ylabel("number users")
plt.title("Evolution of number users")
plt.show()


# In[86]:


Year=[2015, 2017, 2019, 2021, 2023, 2025]
N_views=[1234, 2345, 8224,9345, 2765, 3456]
N_Followers=[567, 1234, 7089, 8456, 1235, 2000]


# In[87]:


plt.figure(figsize=(8, 5))
plt.plot(Year, N_views, label="views")
plt.plot(Year, N_Followers, label="Followers")
plt.ylabel("N Views and Followers")
plt.legend()
plt.show()


# In[90]:


# Point Clouds (scatter plot)
plt.figure(figsize=(5,4))
plt.scatter( N_views,N_Followers)
plt.xlabel('views')
plt.ylabel("followers")
plt.show()


# In[92]:


import plotly

import plotly.graph_objects as go

import plotly.offline as pyo

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

fig = go.Figure(data=go.Scatter(x=year, y=number_users, mode='markers'))

fig.show()


# In[ ]:




