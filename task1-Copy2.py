#!/usr/bin/env python
# coding: utf-8

# In[16]:


#importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[17]:


#data reading from given data link
url="http://bit.ly/w-data"
s_data=pd.read_csv(url)
print("Data imported successful")


s_data.head(10)


# In[18]:


#plotting the distribution of scores
s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[19]:


#splitting of data into inputs and output labels
X=s_data.iloc[:, :-1].values
Y=s_data.iloc[:, 1].values


# In[20]:


#splitting data into training,testing sets,training the algorithm
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1),Y_train)

print("Training Completed")


# In[21]:


#visualizing the best-fit line of regression
line= regressor.coef_*X+regressor.intercept_

#plotting for the test data
plt.scatter(X,Y)
plt.plot(X,line,color='black');
plt.show()


# In[22]:


#testing data
print(X_test)


# In[23]:


#model prediction
Y_pred=regressor.predict(X_test)


# In[24]:


#comparing actual vs predicted
df=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
df


# In[25]:


#estimating training and test score
print("Training Score:",regressor.score(X_train,Y_train))
print("Test Score:",regressor.score(X_test,Y_test))


# In[26]:


#plotting through bar graph to predict the difference between the actual and predicted values

df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major',linewidth='0.5',color='red')
plt.grid(which='minor',linewidth='0.5',color='blue')
plt.show()


# In[27]:


#Testing the model with our own data
hours =  9.25
test = np.array([hours])
test = test.reshape(-1,1)
own_pred=regressor.predict(test)
print("No of hours={}".format(hours))
print("Predicted Score={}".format(own_pred[0]))


# In[29]:


#evaluating the model
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,Y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
print('R-2:',metrics.r2_score(Y_test,Y_pred))


# In[ ]:




