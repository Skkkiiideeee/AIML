# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:07:25 2024

@author: KIIT
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bc = pd.read_csv(r"C:\Users\KIIT\Downloads\archive (3)\data.csv")
bc

bc.info()

bc.describe()

def tochecktoken(k,Sheet):
    for i in Sheet:
        print(i,'=',sum(Sheet[i]==k))
        
def tocheckunique(Sheet):
    for i in Sheet:
        print(i,'\n')
        print(Sheet[i].unique(),'\n')

tocheckunique(bc)

bc.isnull().sum()

import warnings
warnings.filterwarnings(action="ignore")

sns.displot(bc.diagnosis)
plt.show()

bc1=bc.drop("diagnosis",axis=1)
cor=bc1.corr()
cor

sns.heatmap(cor,annot=True)
plt.show()

bc.columns

ip=bc.drop(['diagnosis','id'],axis=1)
ip

op=bc['diagnosis']
op

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

op=le.fit_transform(op)
op

from keras.utils import to_categorical
op=to_categorical(op,2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ip,op,train_size=0.8)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model=Sequential()
#input layer
model.add(Dense(90,input_dim=30,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(45,activation='relu'))
#output layer
model.add(Dense(2,activation='softmax'))
model.compile(Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

h=model.fit(x_train, y_train, epochs=20,validation_data=(x_test,y_test))

y_pred = model.predict(x_test)

c=[]
for i in y_pred:
  c.append(np.argmax(i))
print(c)

c1 = []
for i in y_test:
  c1.append(np.argmax(i))
print(c1)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

acc = accuracy_score(c,c1)
pre = precision_score(c,c1,average='macro')
recall = recall_score(c,c1,average='macro')
f1 = f1_score(c,c1,average='macro')
print(acc,pre,recall,f1)