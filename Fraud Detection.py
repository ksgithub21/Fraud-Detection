#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[35]:


#loading the data 
data = pd.read_csv(r"C:\Users\hp\Downloads\archive\creditcard.csv")


# In[51]:


# first 5 rows
data.head()


# In[52]:


data.tail()


# In[110]:


#data.columns


# In[111]:


data.info()


# In[112]:


data.isnull().sum()


# In[113]:


data['Class'].value_counts()


# In[106]:


#0--> Normal Transaction
#1--> Fraudlent Transaction


# In[114]:


#seperating the data for analysis
legit = data[data.Class==0]
fraud = data[data.Class==1]


# In[115]:


print(legit.shape)


# In[116]:


print(fraud.shape)


# In[117]:


legit.Amount.describe()


# In[118]:


fraud.Amount.describe()


# In[119]:


data.groupby('Class').mean()


# In[120]:


#Build sample dataset containing similar distribution of naormal transactions and fraudlent transaction


# In[121]:


legit_sample = legit.sample(n=492)


# In[122]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)


# In[123]:


new_dataset.head()


# In[124]:


new_dataset.tail()


# In[125]:


new_dataset['Class'].value_counts()


# In[126]:


new_dataset.groupby('Class').mean()


# In[128]:


x= new_dataset.drop(columns='Class',axis=1)
y= new_dataset['Class']


# In[129]:


print(x)


# In[130]:


print(y)


# In[131]:


#split data in to training and testing data


# In[132]:


x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)


# In[133]:


print(x.shape,x_train.shape,x_test.shape)


# In[134]:


#Model Training
# Logistic Regression


# In[135]:


model = LogisticRegression()


# In[136]:


#Training LRModel with Training Data


# In[137]:


model.fit(x_train, y_train)


# In[138]:


#Model Evluation
#Accuracy Score


# In[143]:


x_train_prediction= model.predict(x_train)
training_data_accuracy =  accuracy_score(x_train_prediction, y_train)


# In[145]:


print('Accuracy on Training data :',training_data_accuracy)


# In[146]:


#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[149]:


print('Accuracy score on Test Data:',test_data_accuracy)

