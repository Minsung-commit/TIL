#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("online_courses_2014_2016.csv")
df.head(2)


# ### [[ 문제 1 ]]

# In[3]:


df["income"] = df["price"] * df["subscribers"]
df["review_rate"] = df["reviews"] / df["subscribers"]
df.head(2)


# In[4]:


sum((df["income"] >= 10000) & (df["review_rate"] >= 0.1))


# ### [[ 문제 2 ]]

# In[5]:


df["published"] = pd.to_datetime(df["published"])
df["year"] = df["published"].dt.year
df.head(2)


# In[6]:


df_sub = df.loc[(df["year"] == 2016) & (df["subject"] == "Web Development"), ]
df_sub.head(2)


# In[7]:


df_sub[["price", "subscribers"]].corr()


# In[8]:


round(0.034392, 2)


# ### [[ 문제 3 ]]

# In[9]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# In[10]:


model = ols(formula = "review_rate ~ C(year)", data = df).fit()
anova_lm(model)


# In[11]:


round(18.542038, 1)


# In[ ]:




