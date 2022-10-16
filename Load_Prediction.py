#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
A Company wants to automate the loan eligibility process based on customer details provided while filling online application form. The details filled by the customer are Gender, Marital Status, Education, Number of Dependents, Income of self and co applicant, Required Loan Amount, Required Loan Term, Credit History and others. The requirements are as follows:

Check eligibility of the Customer given the inputs described above.
# # Filter Warnings 

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# # Read Dataset

# In[2]:


import pandas as pd


# In[3]:


train = pd.read_csv("F:/Course/ML/DL/18_Project/training_set (2).csv")
test = pd.read_csv("F:/Course/ML/DL/18_Project/testing_set (1).csv")


# In[4]:


train.head(5)


# In[5]:


test.head(5)


# In[6]:


train.isna().sum()


# In[7]:


test.isna().sum()


# # Missing data treatment

# In[8]:


from ml_functions import replacer,chisq,outliers,preprocessing,standardize,catconsep,ANOVA


# In[9]:


replacer(train)
replacer(test)


# In[10]:


train.isna().sum()


# In[11]:


test.isna().sum()


# # Finding unsual data in data set

# In[12]:


for i in train.columns:
    print(train[i].value_counts())


# Only column "Dependents" have a unsual value of 3+ 

# In[13]:


for i in test.columns:
   print(test[i].value_counts())


# Only column "Dependents" have a unsual value of 3+

# # Replacing unsual values from both dataset

# In[14]:


train["Dependents"] = train["Dependents"].replace(["3+"], 4)
test["Dependents"] = test["Dependents"].replace(["3+"], 4)


# In[15]:


train["Dependents"].value_counts()


# In[16]:


test["Dependents"].value_counts()


# # Define X and Y

# In[17]:


Y = train["Loan_Status"]
X = train.drop(labels=["Loan_ID","Loan_Status"],axis = 1)


# In[18]:


X


# In[19]:


Y


# In[20]:


cat,con = catconsep(X)


# In[21]:


Xnew = preprocessing(X)


# # Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(Xnew,Y,test_size=0.2,random_state =21)


# # First model using Decision Tree
# DTC model with max_depth  
# In[23]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier


for i in range(2,20,1): 
    dtc = DecisionTreeClassifier(criterion="entropy", random_state = 21, max_depth = i)
    model1 = dtc.fit(xtrain,ytrain)
    pred_tr1 = model1.predict(xtrain)
    pred_ts1 = model1.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr1_acc = accuracy_score(ytrain,pred_tr1)
    ts1_acc = accuracy_score(ytest,pred_ts1)
    tr.append(tr1_acc)
    ts.append(ts1_acc)


import matplotlib.pyplot as plt 
plt.plot(range(2,20,1),tr,c="blue")
plt.plot(range(2,20,1),ts,c="red")

#DTC for min_samples_leaf
# In[24]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier


for i in range(2,40,1): 
    dtc = DecisionTreeClassifier(criterion="entropy", random_state = 21, min_samples_leaf = i)
    model1 = dtc.fit(xtrain,ytrain)
    pred_tr1 = model1.predict(xtrain)
    pred_ts1 = model1.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr1_acc = accuracy_score(ytrain,pred_tr1)
    ts1_acc = accuracy_score(ytest,pred_ts1)
    tr.append(tr1_acc)
    ts.append(ts1_acc)


import matplotlib.pyplot as plt 
plt.plot(range(2,40,1),tr,c="blue")
plt.plot(range(2,40,1),ts,c="red")

#DTC for min_samples_split
# In[25]:


tr = []
ts = []
from sklearn.tree import DecisionTreeClassifier


for i in range(2,80,1): 
    dtc = DecisionTreeClassifier(criterion="entropy", random_state = 21, min_samples_split = i)
    model1 = dtc.fit(xtrain,ytrain)
    pred_tr1 = model1.predict(xtrain)
    pred_ts1 = model1.predict(xtest)

    from sklearn.metrics import accuracy_score
    tr1_acc = accuracy_score(ytrain,pred_tr1)
    ts1_acc = accuracy_score(ytest,pred_ts1)
    tr.append(tr1_acc)
    ts.append(ts1_acc)


import matplotlib.pyplot as plt 
plt.plot(range(2,80,1),tr,c="blue")
plt.plot(range(2,80,1),ts,c="red")

# DTC using max_depth as final model for DTC
# In[26]:


dtc = DecisionTreeClassifier(criterion="entropy", random_state = 21, max_depth = 4)
model1 = dtc.fit(xtrain,ytrain)
pred_tr1 = model1.predict(xtrain)
pred_ts1 = model1.predict(xtest)

from sklearn.metrics import accuracy_score
tr1_acc = accuracy_score(ytrain,pred_tr1)
ts1_acc = accuracy_score(ytest,pred_ts1)               


# In[27]:


tr1_acc


# In[28]:


ts1_acc


# In[29]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytrain,pred_tr1))
print(confusion_matrix(ytest,pred_ts1))


# # Second model using RandomForest
#RFC with max depth
# In[30]:


tr1 = []
ts1 = []

for  i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20,criterion="entropy",random_state=21,max_depth=i)
    model2 = rfc.fit(xtrain,ytrain)
    pred_tr2 = model2.predict(xtrain)
    pred_ts2 = model2.predict(xtest)
    tr2_acc = accuracy_score(ytrain,pred_tr2)
    ts2_acc = accuracy_score(ytest,pred_ts2)
    tr1.append(tr2_acc)
    ts1.append(ts2_acc)

plt.plot(range(2,20,1),tr1,c="blue")
plt.plot(range(2,20,1),ts1,c="red")

#RFC with min_samples_leaf
# In[31]:


tr1 = []
ts1 = []

for  i in range(2,20,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20, criterion="entropy",random_state=21,min_samples_leaf=i)
    model2 = rfc.fit(xtrain,ytrain)
    pred_tr2 = model2.predict(xtrain)
    pred_ts2 = model2.predict(xtest)
    tr2_acc = accuracy_score(ytrain,pred_tr2)
    ts2_acc = accuracy_score(ytest,pred_ts2)
    tr1.append(tr2_acc)
    ts1.append(ts2_acc)

plt.plot(range(2,20,1),tr1,c="blue")
plt.plot(range(2,20,1),ts1,c="red")

#RFC with min_samples_split
# In[32]:


tr1 = []
ts1 = []

for  i in range(2,70,1):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20, criterion="entropy",random_state=21,min_samples_split=i)
    model2 = rfc.fit(xtrain,ytrain)
    pred_tr2 = model2.predict(xtrain)
    pred_ts2 = model2.predict(xtest)
    tr2_acc = accuracy_score(ytrain,pred_tr2)
    ts2_acc = accuracy_score(ytest,pred_ts2)
    tr1.append(tr2_acc)
    ts1.append(ts2_acc)

plt.plot(range(2,70,1),tr1,c="blue")
plt.plot(range(2,70,1),ts1,c="red")

#RFC final model with max_depth
# In[33]:


rfc = RandomForestClassifier(n_estimators=20,criterion="entropy",random_state=21,max_depth=5)
model2 = rfc.fit(xtrain,ytrain)
pred_tr2 = model2.predict(xtrain)
pred_ts2 = model2.predict(xtest)
tr2_acc = accuracy_score(ytrain,pred_tr2)
ts2_acc = accuracy_score(ytest,pred_ts2)


# In[34]:


print(tr2_acc)


# In[35]:


print(ts2_acc)


# In[36]:


print(confusion_matrix(ytrain,pred_tr2))
print(confusion_matrix(ytest,pred_ts2))


# # Thrid model using Logistic Reression

# In[37]:


imp_cols = []
pvals = []

for i in X.columns:
    if (X[i].dtypes == "object"):
        pval = chisq(train,"Loan_Status",i)
        pvals.append(pval)
    else:
        pval = ANOVA(train,"Loan_Status",i)
        pvals.append(pval)


# In[38]:


W = pd.DataFrame([X.columns,pvals]).T


# In[39]:


W.columns=("col","pval")


# In[40]:


W


# In[41]:


W[W.pval<0.05]


# In[42]:


X1 = X


# In[43]:


X1 =X1.drop(labels=["Gender","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"],axis=1)


# In[44]:


X1


# In[45]:


X1new =preprocessing(X1)


# In[46]:


xtrain,xtest,ytrain,ytest =train_test_split(X1new,Y,random_state=21,test_size=0.2)


# In[47]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model3 =lr.fit(xtrain,ytrain)
pred_tr3 = model3.predict(xtrain)
pred_ts3 = model3.predict(xtest)


# In[48]:


print(round(accuracy_score(ytrain,pred_tr3),4))
print(round(accuracy_score(ytest,pred_ts3),4))


# In[49]:


print(confusion_matrix(ytrain,pred_tr3))
print(confusion_matrix(ytest,pred_ts3))


# In[50]:


pred_ts3


# # Fourth model using Naive Bayes

# In[51]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
model4 = nb.fit(xtrain,ytrain)
pred_tr4 = model4.predict(xtrain)
pred_ts4 = model4.predict(xtest)

tr4_acc = accuracy_score(ytrain,pred_tr4)
ts4_acc = accuracy_score(ytest,pred_ts4)

print("Training Accuracy : ", round(tr4_acc,4))
print("Testing Accuracy : ", round(ts4_acc,4))

# For all the model tested the best model we got is from DTC with max_depth
# Selecting DTC with max_depth as final model for prediction 
# # Preparing Test data

# In[52]:


Xtest = test.drop(labels="Loan_ID", axis = 1)
cat_test,con_test=catconsep(Xtest)
Xtest_new = preprocessing(Xtest)


# In[53]:


Xtest_new


# In[54]:


pred_test = model1.predict(Xtest_new)


# In[55]:


pred_test


# In[56]:


T = test[["Loan_ID"]]
T["Loan_Status"] = pred_test


# In[57]:


T


# In[ ]:


#T.to_csv("Desktop/Loan_Prediction.csv")

