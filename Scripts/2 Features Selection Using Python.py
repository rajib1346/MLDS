#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

width_in_inches=20
height_in_inches=20
dots_in_inch=100

p_data = pd.read_csv("E:/New folder/cardio_Data.CSV")
x = p_data.drop('target', axis=1).values
y = p_data['target'].values

names = p_data.drop('target',axis=1).columns

plt.figure(figsize=(width_in_inches,height_in_inches),dpi=dots_in_inch)

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(x,y).coef_
_=plt.plot(range(len(names)),lasso_coef)
_=plt.xticks(range(len(names)),names, rotation=60)
_=plt.ylabel("Coefficients")
plt.show()


# In[10]:


print(lasso.coef_)


# In[ ]:





# In[1]:


import pandas as pd
import numpy as np

frame=pd.read_csv("E:/New folder/cardio_Data.CSV")
x = frame[['age','height','gender','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y = frame[['target']]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

