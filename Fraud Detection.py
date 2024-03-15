import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\hp\Downloads\archive\creditcard.csv")
data.head()
data.tail()
data.columns
data.info()
data.isnull().sum()
data['Class'].value_counts()
legit = data[data.Class==0]
fraud = data[data.Class==1]
print(legit.shape)
print(fraud.shape)
legit.Amount.describe()
fraud.Amount.describe()
data.groupby('Class').mean()
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample,fraud],axis=0)
new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()
x= new_dataset.drop(columns='Class',axis=1)
y= new_dataset['Class']
print(x)
print(y)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)
print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
x_train_prediction= model.predict(x_train)
training_data_accuracy =  accuracy_score(x_train_prediction, y_train)
print('Accuracy on Training data :',training_data_accuracy)
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print('Accuracy score on Test Data:',test_data_accuracy)
