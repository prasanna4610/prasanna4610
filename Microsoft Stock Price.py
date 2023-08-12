#!/usr/bin/env python
# coding: utf-8

# In[7]:


from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns


# In[8]:


df=pd.read_csv("Microsoft stock price.csv")
print(df)


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# In[15]:


plt.bar(df['Date'],df['Open'],color="red",label="Open")
plt.bar(df['Date'],df['Close'],color="blue",label="Close")
plt.title("Microsoft Open & Close Stock")
plt.legend()


# In[16]:


plt.plot(df['Date'],df['Volume'],color="green",marker='*',markerfacecolor='yellow')
plt.show()


# In[17]:


print(df.corr())
plt.show()


# In[18]:


sns.heatmap(df.corr(),annot=True,cbar=False,cmap = "tab20")
plt.show()


# In[30]:


df['Date'] = pd.to_datetime(df['Date'])
prediction = df.loc[(df['Date']> datetime(2013, 1, 1))& (df['Date']< datetime(2018, 1, 1))]
plt.figure(figsize=(10, 10))
plt.bar(df['Date'], df['Close'],color='#f000ff')
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Microsoft Stock Prices")


# In[21]:


x = df[["Open", "High", "Low"]]
y = df["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[27]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Rate": ypred})
print(data.head())

