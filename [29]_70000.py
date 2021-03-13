#!/usr/bin/env python
# coding: utf-8

# In[211]:


#features selection using Random search N-1
import pandas as pd
import numpy as np 
all_features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11"]
boolean_mask1  =[1]
boolean_mask2  =[1,1]
boolean_mask3  =[1,1,1]
boolean_mask4  =[1,1,1,1]
boolean_mask5  =[1,1,1,1,1]
boolean_mask6  =[1,1,1,1,1,1]
boolean_mask7  =[1,1,1,1,1,1,1]
boolean_mask8  =[1,1,1,1,1,1,1,1]
boolean_mask9  =[1,1,1,1,1,1,1,1,1]
boolean_mask10 =[1,1,1,1,1,1,1,1,1,1]

features1=np.random.choice(all_features,size=1,replace=False)
features2=np.random.choice(all_features,size=2,replace=False)
features3=np.random.choice(all_features,size=3,replace=False)
features4=np.random.choice(all_features,size=4,replace=False)
features5=np.random.choice(all_features,size=5,replace=False)
features6=np.random.choice(all_features,size=6,replace=False)
features7=np.random.choice(all_features,size=7,replace=False)
features8=np.random.choice(all_features,size=8,replace=False)
features9=np.random.choice(all_features,size=9,replace=False)
features10=np.random.choice(all_features,size=10,replace=False)

df1= pd.DataFrame(features1,boolean_mask1)
df2 = pd.DataFrame(features2,boolean_mask2)
df3 = pd.DataFrame(features3,boolean_mask3)
df4 = pd.DataFrame(features4,boolean_mask4)
df5 = pd.DataFrame(features5,boolean_mask5)
df6 = pd.DataFrame(features6,boolean_mask6)
df7 = pd.DataFrame(features7,boolean_mask7)
df8 = pd.DataFrame(features8,boolean_mask8)
df9 = pd.DataFrame(features9,boolean_mask9)
df10 = pd.DataFrame(features10,boolean_mask10)

print('Number of features 1\n',df1)
print('Number of features 2\n',df2)
print('Number of features 3\n',df3)
print('Number of features 4\n',df4)
print('Number of features 5\n',df5)
print('Number of features 6\n',df6)
print('Number of features 7\n',df7)
print('Number of features 8\n',df8)
print('Number of features 9\n',df9)
print('Number of features 10\n',df10)


# In[264]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv("C:/Users/acer/Downloads/22/RSA-RF_(70000)/Initial_Train.CSV")
test=pd.read_csv("C:/Users/acer/Downloads/22/RSA-RF_(70000)/Initial_Test.CSV")

x_train1 =train[['A10']]
y_train1 = train[['target']]
x_test1 = test[['A10']]
y_test1 = test[['target']]

x_train2 =train[['A2','A5']]
y_train2 = train[['target']]
x_test2 = test[['A2','A5']]
y_test2 = test[['target']]

x_train3 =train[['A6','A10','A4']]
y_train3 = train[['target']]
x_test3 = test[['A6','A10','A4']]
y_test3 = test[['target']]

x_train4 =train[['A8','A11','A4','A3']]
y_train4 = train[['target']]
x_test4 = test[['A8','A11','A4','A3']]
y_test4 = test[['target']]

x_train5 =train[['A3','A11','A5','A10','A8']]
y_train5 = train[['target']]
x_test5 = test[['A3','A11','A5','A10','A8']]
y_test5 = test[['target']]

x_train6 =train[['A9','A8','A1','A6','A3','A7']]
y_train6 = train[['target']]
x_test6 = test[['A9','A8','A1','A6','A3','A7']]
y_test6 = test[['target']]

x_train7 =train[['A9','A2','A3','A6','A7','A4','A4']]
y_train7 = train[['target']]
x_test7 = test[['A9','A2','A3','A6','A7','A4','A4']]
y_test7 = test[['target']]

x_train8 =train[['A7','A10','A5','A3','A1','A11','A8','A2']]
y_train8 = train[['target']]
x_test8 = test[['A7','A10','A5','A3','A1','A11','A8','A2']]
y_test8 = test[['target']]

x_train9 =train[['A1','A8','A10','A2','A5','A4','A6','A3','A11']]
y_train9 = train[['target']]
x_test9 = test[['A1','A8','A10','A2','A5','A4','A6','A3','A11']]
y_test9 = test[['target']]

x_train10 =train[['A8','A6','A4','A2','A3','A5','A11','A10','A7','A9']]
y_train10 = train[['target']]
x_test10 = test[['A8','A6','A4','A2','A3','A5','A11','A10','A7','A9']]
y_test10 = test[['target']]


# In[267]:


random_forest1=RandomForestClassifier(n_estimators=1,max_depth=2)
random_forest1.fit(x_train1,y_train1)
rf_1=random_forest1.predict(x_test1)
acc_rf_1=metrics.accuracy_score(y_test1,rf_1)*100
print('Accuracu_1= \n',acc_rf_1)

random_forest2=RandomForestClassifier(n_estimators=2,max_depth=2)
random_forest2.fit(x_train2,y_train2)
rf_2=random_forest2.predict(x_test2)
acc_rf_2=metrics.accuracy_score(y_test2,rf_2)*100
print('Accuracu_2= \n',acc_rf_2)

random_forest3=RandomForestClassifier(n_estimators=8,max_depth=2)
random_forest3.fit(x_train3,y_train3)
rf_3=random_forest3.predict(x_test3)
acc_rf_3=metrics.accuracy_score(y_test3,rf_3)*100
print('Accuracu_3= \n',acc_rf_3)

random_forest4=RandomForestClassifier(n_estimators=8,max_depth=2)
random_forest4.fit(x_train4,y_train4)
rf_4=random_forest4.predict(x_test4)
acc_rf_4=metrics.accuracy_score(y_test4,rf_4)*100
print('Accuracu_4= \n',acc_rf_4)

random_forest5=RandomForestClassifier(n_estimators=92,max_depth=2)
random_forest5.fit(x_train5,y_train5)
rf_5=random_forest5.predict(x_test5)
acc_rf_5=metrics.accuracy_score(y_test5,rf_5)*100
print('Accuracu_5= \n',acc_rf_5)

random_forest6=RandomForestClassifier(n_estimators=92,max_depth=12)
random_forest6.fit(x_train6,y_train6)
rf_6=random_forest6.predict(x_test6)
acc_rf_6=metrics.accuracy_score(y_test6,rf_6)*100
print('Accuracu_6= \n',acc_rf_6)

random_forest7=RandomForestClassifier(n_estimators=73,max_depth=4)
random_forest7.fit(x_train7,y_train7)
rf_7=random_forest7.predict(x_test7)
acc_rf_7=metrics.accuracy_score(y_test7,rf_7)*100
print('Accuracu_7= \n',acc_rf_7)

random_forest8=RandomForestClassifier(n_estimators=95,max_depth=2)
random_forest8.fit(x_train8,y_train8)
rf_8=random_forest8.predict(x_test8)
acc_rf_8=metrics.accuracy_score(y_test8,rf_8)*100
print('Accuracu_8= \n',acc_rf_8)

random_forest9=RandomForestClassifier(n_estimators=95,max_depth=2)
random_forest9.fit(x_train9,y_train9)
rf_9=random_forest9.predict(x_test9)
acc_rf_9=metrics.accuracy_score(y_test9,rf_9)*100
print('Accuracu_9= \n',acc_rf_9)

random_forest10=RandomForestClassifier(n_estimators=95,max_depth=2)
random_forest10.fit(x_train10,y_train10)
rf_10=random_forest10.predict(x_test10)
acc_rf_10=metrics.accuracy_score(y_test10,rf_10)*100
print('Accuracu_10= \n',acc_rf_10)

