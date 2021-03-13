#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("E:/New folder/cardio_Data.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('E:/New folder/Initial_Train.csv')
x_test.to_csv('E:/New folder/Initial_Test.csv')


# In[2]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("E:/cardio_Data.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('E:/Initial_Train.csv')
x_test.to_csv('E:/Initial_Test.csv')


# In[2]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("E:/cardio_Data.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('E:/Initial_Train.csv')
x_test.to_csv('E:/Initial_Test.csv')


# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("E:/cardio_Data.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.5,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('E:/Initial_Train.csv')
x_test.to_csv('E:/Initial_Test.csv')


# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Data Set spilting
DataFrame=pd.read_csv("E:/cardio_Data.CSV")

x = DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = DataFrame[['target']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4,random_state=5)

x_train['target']=y_train
x_test['target']=y_test

x_train.to_csv('E:/Initial_Train.csv')
x_test.to_csv('E:/Initial_Test.csv')


# In[5]:


import pandas as pd 
DataFrame=pd.read_csv("E:/cardio_Data.CSV")
DataFrame.isnull().sum()


# In[ ]:




