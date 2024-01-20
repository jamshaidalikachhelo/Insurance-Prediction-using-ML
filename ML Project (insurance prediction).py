#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('E:\ML PROJECTS\insurance_data.csv')
df


# In[3]:


df.head(3)


# In[4]:


plt.scatter(df.age,df.bought_insurance,marker='+', color = 'red')


# In[5]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.2)


# In[6]:


X_test


# In[7]:


X_train


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


reg = LogisticRegression()


# In[10]:


reg.fit(X_train,y_train)


# In[11]:


reg.score(X_test,y_test)


# In[12]:


reg.predict(X_test)


# In[ ]:





# In[ ]:




