#!/usr/bin/env python
# coding: utf-8

# # storm motor wants to develop an algo. to predict the price of the cars based on various attributes associated with the car

# In[38]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[39]:


sns.set(rc={'figure.figsize':(10,7)})


# In[40]:


car_data=pd.read_csv('C:\\Users\\pradeep\\Desktop\\dataset python\\Data science with python dataset\\cars.csv')


# In[41]:


car=car_data.copy()


# In[42]:


car.info()


# In[43]:


car.describe()


# # to display maximum set of columns

# In[44]:


pd.set_option('display.max_columns',100)
car.describe()


# # dropping unwanted columns

# In[45]:


col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
car=car.drop(col,axis=1)


# # removing duplicate record

# In[46]:


car.drop_duplicates(keep='first',inplace=True)


# # data cleaning

# In[47]:


car.isnull().sum()


# In[48]:


# variable year of registration


# In[49]:


yearwise_count=car['yearOfRegistration'].value_counts().sort_index()
yearwise_count


# In[50]:


# working range 1950 to 2018


# In[51]:


sum(car['yearOfRegistration']>2018)
sum(car['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=car)


# In[52]:


# variable price


# In[53]:


price_count=car['price'].value_counts().sort_index()
sns.distplot(car['price'])
car['price'].describe()
sns.boxplot(y=car['price'])


# In[54]:


# working range 100 and 150000


# In[55]:


sum(car['price']>150000)
sum(car['price']<500)


# In[56]:


# variable powerPS


# In[57]:


power_count=car['powerPS'].value_counts().sort_index()
sns.distplot(car['powerPS'])
car['powerPS'].describe()
sns.boxplot(y=car['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=car)


# In[58]:


# working range 10 and 500


# In[59]:


sum(car['powerPS']>500)
sum(car['powerPS']<10)


# # working range of data

# In[60]:


# 6700 record are dropped using below code


# In[61]:


car=car[(car.yearOfRegistration <=2018)
        & (car.yearOfRegistration >=1950)
        & (car.price >=100)
        & (car.price <=150000)
        & (car.powerPS >=10)
        & (car.powerPS <=500)
       ]


# In[62]:


# Further More simplify variable by reducing 


# In[63]:


# combining yearofregistration and monthofregistration


# In[64]:


car['monthOfRegistration']/=12


# In[65]:


# creating age by adding yearOfRegistration and monthOfRegistration


# In[66]:


car['age']=(2018-car['yearOfRegistration']) + car['monthOfRegistration']
car['age']=round(car['age'],2)
car['age'].describe()


# In[67]:


# dropping yearOfRegistration and monthOfRegistration


# In[68]:


car=car.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
print(car)


# # Visualization

# In[69]:


# age


# In[72]:


sns.distplot(car['age'])


# In[ ]:


sns.boxplot(y=car['age'])


# In[ ]:


# price


# In[73]:


sns.distplot(car['price'])


# In[75]:


sns.boxplot(y=car['price'])


# In[78]:


# powerPS


# In[79]:


sns.distplot(car['powerPS'])


# In[80]:


sns.boxplot(y=car['powerPS'])


# In[76]:


# age vs price


# In[77]:


sns.regplot(x='age',y='price',scatter=True,fit_reg=False,data=car)


# In[81]:


# powerPS VS price


# In[86]:


sns.regplot(x=car['powerPS'],y=car['price'],scatter=True,fit_reg=False,data=car)


# In[ ]:


# Find the insignificant variable, after finding insignificant variable we will remove them.


# In[87]:


# variable  seller(categorical var.)


# In[88]:


car['seller'].value_counts()
pd.crosstab(car['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=car)

# Fewer cars have commercial= insignificant


# In[89]:


# variable offerType


# In[90]:


car['offerType'].value_counts()
pd.crosstab(car['offerType'],columns='count',normalize=True)
sns.countplot(x='offerType',data=car)

# all cars have offer=insignificant


# In[91]:


# variable abtest


# In[93]:


car['abtest'].value_counts()
pd.crosstab(car['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=car)

# equally distributed =insignificant


# In[94]:


sns.boxplot(x='abtest',y='price',data=car)
# for every price value there is almost 50:50 distribution
# does not affect price= insignificant


# In[95]:


# variable vehicleType


# In[99]:


car['vehicleType'].value_counts()
pd.crosstab(car['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=car)


# In[98]:


sns.boxplot(x='vehicleType',y='price',data=car)

# vehicleType affects price


# In[100]:


# variable gearbox


# In[102]:


car['gearbox'].value_counts()
pd.crosstab(car['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=car)


# In[103]:


sns.boxplot(x='gearbox',y='price',data=car)

# gearbox affects price


# In[104]:


# variable model


# In[105]:


car['model'].value_counts()
pd.crosstab(car['model'],columns='count',normalize=True)
sns.countplot(x='model',data=car)


# In[107]:


sns.boxplot(x='model',y='price',data=car)
# cars are distributed over many models
# significant variable


# In[108]:


# Variable kilometer


# In[109]:


car['kilometer'].value_counts().sort_index()
pd.crosstab(car['kilometer'],columns='count',normalize=True)
sns.boxplot(x='kilometer',y='price',data=car)
car['kilometer'].describe()


# In[110]:


sns.distplot(car['kilometer'],bins=8,kde=False)


# In[111]:


sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=car)
# significant variable


# In[112]:


# variable fuelType


# In[113]:


car['fuelType'].value_counts()
pd.crosstab(car['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=car)


# In[114]:


sns.boxplot(x='fuelType',y='price',data=car)
# affects price=significant


# In[115]:


# variable brand


# In[116]:


car['brand'].value_counts()
pd.crosstab(car['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=car)


# In[117]:


sns.boxplot(x='brand',y='price',data=car)
# cars are distributed over many brands= significant


# # Removing insignificant variables

# In[118]:


col=['seller','offerType','abtest']
car=car.drop(columns=col,axis=1)
car1=car.copy()


# In[119]:


# correlation 


# In[120]:


car_cor=car1.select_dtypes(exclude=[object])
car_cor.corr()


# In[122]:


car_cor.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
# highest correlation is between powerPS with price


# In[ ]:





# In[ ]:




