#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[111]:


DataFrame=pd.read_csv("C:/Users/acer/Downloads/20/cardio_train.CSV")
DataFrame1=pd.read_csv("C:/Users/acer/Downloads/20/Initial_Train.CSV")
DataFrame2=pd.read_csv("C:/Users/acer/Downloads/20/Initial_Test.CSV")


# In[112]:


x= DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y= DataFrame['target']
x_train = DataFrame1[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_train = DataFrame1['target']
x_test = DataFrame2[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y_test = DataFrame2['target']


# In[113]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
MLP_FdS=MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5,4), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
MLP_FdS.fit(x_train,y_train)
pred_MLP_FdS=MLP_FdS.predict(x_test)
print('Accuracy MLP_FdS=',accuracy_score(y_test,pred_MLP_FdS)*100)

from sklearn import tree
DT_FdS=tree.DecisionTreeClassifier(max_depth=5)
DT_FdS.fit(x_train,y_train)
pred_DT_FdS=DT_FdS.predict(x_test)
print('Accuracy DT_FdS=',accuracy_score(y_test,pred_DT_FdS)*100)

RF_FdS=RandomForestClassifier()
RF_FdS.fit(x_train,y_train)
pred_RF_FdS=RF_FdS.predict(x_test)
print('Accuracy RF_FdS=',accuracy_score(y_test,pred_RF_FdS)*100)

NV_FdS=GaussianNB()
NV_FdS.fit(x_train,y_train)
pred_NV_FdS=NV_FdS.predict(x_test)
print('Accuracy NV_FdS=',accuracy_score(y_test,pred_NV_FdS)*100)

GB_FdS=GradientBoostingClassifier()
GB_FdS.fit(x_train,y_train)
pred_GB_FdS=GB_FdS.predict(x_test)
print('Accuracy GB_FdS=',accuracy_score(y_test,pred_GB_FdS)*100)

LR_FdS=LogisticRegression()
LR_FdS.fit(x_train,y_train)
pred_LR_FdS=LR_FdS.predict(x_test)
print('Accuracy LR_FdS=',accuracy_score(y_test,pred_LR_FdS)*100)


# In[114]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
X=MinMaxScaler().fit_transform(x)
chi_scores = chi2(X,y)
chi_scores
p_values = pd.Series(chi_scores[1],index = x.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()


# In[115]:


x_train_chi_2 = DataFrame1[['gender','height','ap_hi','ap_lo','alco']]
y_train_chi_2 = DataFrame1['target']
x_test_chi_2 = DataFrame2[['gender','height','ap_hi','ap_lo','alco']]
y_test_chi_2 = DataFrame2['target']

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
MLP_FdS=MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5,4), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
MLP_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_MLP_FdS=MLP_FdS.predict(x_test_chi_2)
print('Accuracy MLP_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_MLP_FdS)*100)

from sklearn import tree
DT_FdS=tree.DecisionTreeClassifier(max_depth=5)
DT_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_DT_FdS=DT_FdS.predict(x_test_chi_2)
print('Accuracy DT_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_DT_FdS)*100)

RF_FdS=RandomForestClassifier()
RF_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_RF_FdS=RF_FdS.predict(x_test_chi_2)
print('Accuracy RF_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_RF_FdS)*100)

NV_FdS=GaussianNB()
NV_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_NV_FdS=NV_FdS.predict(x_test_chi_2)
print('Accuracy NV_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_NV_FdS)*100)

GB_FdS=GradientBoostingClassifier()
GB_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_GB_FdS=GB_FdS.predict(x_test_chi_2)
print('Accuracy GB_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_GB_FdS)*100)

LR_FdS=LogisticRegression()
LR_FdS.fit(x_train_chi_2,y_train_chi_2)
pred_LR_FdS=LR_FdS.predict(x_test_chi_2)
print('Accuracy LR_FdS_chi_2=',accuracy_score(y_test_chi_2,pred_LR_FdS)*100)


# In[116]:


x = DataFrame[['gender','height','ap_hi','ap_lo','alco']]
y = DataFrame['target']
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)
x_std
features = x_std.T 
covariance_matrix = np.cov(features)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvectors \n%s' %eig_vecs)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvalues \n%s' %eig_vals)


# In[119]:


x_train_chi_2_pca = DataFrame1[['gender','ap_lo']]
y_train_chi_2_pca = DataFrame1['target']
x_test_chi_2_pca = DataFrame2[['gender','ap_lo']]
y_test_chi_2_pca = DataFrame2['target']

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
MLP_FdS=MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5,4), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
MLP_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_MLP_FdS=MLP_FdS.predict(x_test_chi_2_pca)
print('Accuracy MLP_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_MLP_FdS)*100)

from sklearn import tree
DT_FdS=tree.DecisionTreeClassifier(max_depth=5)
DT_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_DT_FdS=DT_FdS.predict(x_test_chi_2_pca)
print('Accuracy DT_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_DT_FdS)*100)

RF_FdS=RandomForestClassifier()
RF_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_RF_FdS=RF_FdS.predict(x_test_chi_2_pca)
print('Accuracy RF_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_RF_FdS)*100)

NV_FdS=GaussianNB()
NV_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_NV_FdS=NV_FdS.predict(x_test_chi_2_pca)
print('Accuracy NV_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_NV_FdS)*100)

GB_FdS=GradientBoostingClassifier()
GB_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_GB_FdS=GB_FdS.predict(x_test_chi_2_pca)
print('Accuracy GB_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_GB_FdS)*100)

LR_FdS=LogisticRegression()
LR_FdS.fit(x_train_chi_2_pca,y_train_chi_2_pca)
pred_LR_FdS=LR_FdS.predict(x_test_chi_2_pca)
print('Accuracy LR_FdS_chi2_pca=',accuracy_score(y_test_chi_2_pca,pred_LR_FdS)*100)


# In[118]:


x= DataFrame[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
y= DataFrame['target']
from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)
x_std
features = x_std.T 
covariance_matrix = np.cov(features)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvectors \n%s' %eig_vecs)
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('Eigenvalues \n%s' %eig_vals)


# In[120]:


x_train_pca = DataFrame1[['age','gender','ap_lo','smoke']]
y_train_pca = DataFrame1['target']
x_test_pca = DataFrame2[['age','gender','ap_lo','smoke']]
y_test_pca = DataFrame2['target']

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
MLP_FdS=MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(5,4), learning_rate='constant',
       learning_rate_init=0.001, max_iter=100, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
MLP_FdS.fit(x_train_pca,y_train_pca)
pred_MLP_FdS=MLP_FdS.predict(x_test_pca)
print('Accuracy MLP_FdS_pca=',accuracy_score(y_test_pca,pred_MLP_FdS)*100)

from sklearn import tree
DT_FdS=tree.DecisionTreeClassifier(max_depth=5)
DT_FdS.fit(x_train_pca,y_train_pca)
pred_DT_FdS=DT_FdS.predict(x_test_pca)
print('Accuracy DT_FdS_pca=',accuracy_score(y_test_pca,pred_DT_FdS)*100)

RF_FdS=RandomForestClassifier()
RF_FdS.fit(x_train_pca,y_train_pca)
pred_RF_FdS=RF_FdS.predict(x_test_pca)
print('Accuracy RF_FdS_pca=',accuracy_score(y_test_pca,pred_RF_FdS)*100)

NV_FdS=GaussianNB()
NV_FdS.fit(x_train_pca,y_train_pca)
pred_NV_FdS=NV_FdS.predict(x_test_pca)
print('Accuracy NV_FdS_pca=',accuracy_score(y_test_pca,pred_NV_FdS)*100)

GB_FdS=GradientBoostingClassifier()
GB_FdS.fit(x_train_pca,y_train_pca)
pred_GB_FdS=GB_FdS.predict(x_test_pca)
print('Accuracy GB_FdS_pca=',accuracy_score(y_test_pca,pred_GB_FdS)*100)

LR_FdS=LogisticRegression()
LR_FdS.fit(x_train_pca,y_train_pca)
pred_LR_FdS=LR_FdS.predict(x_test_pca)
print('Accuracy LR_FdS_pca=',accuracy_score(y_test_pca,pred_LR_FdS)*100)


# In[ ]:




