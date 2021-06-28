#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


df=pd.read_csv("C:/Users/Neethu Santhosh/Desktop/decoder lectures/case study/shopping-data.csv")


# In[3]:


df.head()


# Here we have the following features :
# 1. CustomerID: It is the unique ID given to a customer
# 2. Gender: Gender of the customer
# 3. Age: The age of the customer
# 4. Annual Income(k$): It is the annual income of the customer
# 5. Spending Score: It is the score(out of 100) given to a customer by the mall authorities, based on the money spent and the behavior of the customer.
# 

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df=df.drop_duplicates()


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


a=["Age","Annual Income (k$)","Spending Score (1-100)"]


# In[11]:


for i in a:
    print(df[i].describe())
    print(df[i].skew())
    sns.displot(df[i])
    plt.show()


# In[12]:


sns.countplot(x="Genre",data=df)


# In[13]:


data=df.iloc[:,3:5]


# In[14]:


data.head()


# In[15]:


from sklearn.cluster import KMeans
algo=KMeans(n_clusters=2)
algo.fit(data)


# In[16]:


cen=algo.cluster_centers_


# In[17]:


algo.labels_


# In[18]:


algo.inertia_


# In[19]:


sns.scatterplot(data['Annual Income (k$)'],data['Spending Score (1-100)'],hue=algo.labels_)
sns.scatterplot(cen[:,0],cen[:,1],color='r')


# In[21]:


dis=[]
k=range(1,15)
for i in k:
    algo=KMeans(n_clusters=i)
    algo.fit(data)
    dis.append(algo.inertia_)


# In[22]:


dis


# In[23]:


plt.plot(k,dis)
plt.show()


# In[24]:


algo1=KMeans(n_clusters=5)
algo1.fit(data)


# In[25]:


cen=algo1.cluster_centers_
cen


# In[26]:


sns.scatterplot(data['Annual Income (k$)'],data['Spending Score (1-100)'],hue=algo1.labels_)
sns.scatterplot(cen[:,0],cen[:,1],color='r')


# In[ ]:


Analyzing the Results
We can see that the mall customers can be broadly grouped into 5 groups based on their purchases made in the mall.
In cluster 4(yellow colored) we can see people have low annual income and low spending scores, this is quite reasonable as people having low salaries prefer to buy less, in fact, these are the wise people who know how to spend and save money. The shops/mall will be least interested in people belonging to this cluster.
In cluster 2(blue colored) we can see that people have low income but higher spending scores, these are those people who for some reason love to buy products more often even though they have a low income. Maybe it’s because these people are more than satisfied with the mall services. The shops/malls might not target these people that effectively but still will not lose them.
In cluster 5(pink colored) we see that people have average income and an average spending score, these people again will not be the prime targets of the shops or mall, but again they will be considered and other data analysis techniques may be used to increase their spending score.
In cluster 1(red-colored) we see that people have high income and high spending scores, this is the ideal case for the mall or shops as these people are the prime sources of profit. These people might be the regular customers of the mall and are convinced by the mall’s facilities.
In cluster 3(green colored) we see that people have high income but low spending scores, this is interesting. Maybe these are the people who are unsatisfied or unhappy by the mall’s services. These can be the prime targets of the mall, as they have the potential to spend money. So, the mall authorities will try to add new facilities so that they can attract these people and can meet their needs.
Finally, based on our machine learning technique we may deduce that to increase the profits of the mall, the mall authorities should target people belonging to cluster 3 and cluster 5 and should also maintain its standards to keep the people belonging to cluster 1 and cluster 2 happy and satisfied.


# In[ ]:





# In[ ]:




