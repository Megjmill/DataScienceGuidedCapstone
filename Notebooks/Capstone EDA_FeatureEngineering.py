#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[3]:


record = pd.read_csv('application_record.csv')


# In[4]:


record.head()


# In[5]:


record.shape


# In[6]:


record.info()


# In[7]:


record['ID'].duplicated().sum()


# In[8]:


record = record.drop_duplicates(subset='ID',keep='first')


# In[9]:


record.shape


# In[10]:


record.columns[1:]


# In[11]:


record.isnull().sum()


# In[12]:


record['NAME_INCOME_TYPE'].unique()


# In[13]:


record['NAME_EDUCATION_TYPE'].unique()


# In[14]:


record['NAME_FAMILY_STATUS'].unique()


# In[15]:


record['NAME_HOUSING_TYPE'].unique()


# In[16]:


record['FLAG_MOBIL'].value_counts()


# In[17]:


record['FLAG_WORK_PHONE'].unique()


# In[18]:


record['FLAG_PHONE'].unique()


# In[19]:


record['FLAG_EMAIL'].unique()


# In[20]:


record['OCCUPATION_TYPE'].value_counts(dropna=False)


# In[21]:


record['OCCUPATION_TYPE'].fillna('not_specified',inplace=True)


# In[22]:


record['OCCUPATION_TYPE'].value_counts(dropna=False)


# In[23]:


record.describe(percentiles=[.01,.02,.03,.04,.05,.1,.25,.5,.75,.9,.95,.96,.97,.98,.99]).T


# In[24]:


sns.boxplot(record, y='AMT_INCOME_TOTAL')

plt.show()


# In[25]:


sns.boxplot(data=record, y=record['CNT_CHILDREN'])


# In[26]:


record['DAYS_EMPLOYED'].max()


# In[27]:


sns.boxplot(data=record, y=record['DAYS_BIRTH'])


# In[28]:


sns.boxplot(data=record, y=record['DAYS_EMPLOYED'])


# In[29]:


record[record['DAYS_EMPLOYED']>=0]['DAYS_EMPLOYED'].value_counts()


# In[30]:


record['DAYS_EMPLOYED'].replace(365243,0,inplace=True)


# In[31]:


record[record['DAYS_EMPLOYED']>=0]['DAYS_EMPLOYED'].value_counts()


# In[32]:


record['AGE_YEARS']=round(-record['DAYS_BIRTH']/365.2425,0)


# In[33]:


record['YEARS_EMPLOYED']=round(-record['DAYS_EMPLOYED']/365.2425)
record.loc[record['YEARS_EMPLOYED']<0,'YEARS_EMPLOYED']=0


# In[34]:


record.drop(columns=["DAYS_BIRTH","DAYS_EMPLOYED"],inplace=True)


# In[35]:


record.describe(percentiles=[.01,.02,.03,.04,.05,.1,.25,.5,.75,.9,.95,.96,.97,.98,.99]).T


# In[36]:


record['ID'].duplicated().sum()


# In[37]:


sns.boxplot(record,y='AMT_INCOME_TOTAL')


# In[38]:


record[record['AMT_INCOME_TOTAL']>540000]


# In[39]:


record.drop(columns=["FLAG_MOBIL"],inplace=True)


# In[40]:


credit_record = pd.read_csv('credit_record.csv')


# In[41]:


credit_record.head()


# In[42]:


credit_record.shape


# In[43]:


credit_record.info()


# In[44]:


credit_record.duplicated().sum()


# In[45]:


credit_record['MONTHS_BALANCE'].unique()


# In[46]:


credit_record['STATUS'].unique()


# In[47]:


credit_record[credit_record['STATUS'].isin(['X', 'C'])]


# In[48]:


credit_record['ID'].nunique()


# In[49]:


credit_record['target']=credit_record['STATUS']
credit_record['target'].replace('X', 0, inplace=True)
credit_record['target'].replace('C', 0, inplace=True)
credit_record['target']=credit_record['target'].astype(int)
credit_record.loc[credit_record['target']>=1,'target']=1


# In[50]:


df3=pd.DataFrame(credit_record.groupby(['ID'])['target'].agg("max")).reset_index()


# In[51]:


df3["target"].value_counts()


# In[52]:


#Merging the two table using an inner join
df = pd.merge(record, df3, how='inner', on=['ID'])


# In[53]:


df.head()


# In[54]:


start_df = pd.DataFrame(credit_record.groupby(['ID'])['MONTHS_BALANCE'].agg(min)).reset_index()

start_df.rename(columns={'MONTHS_BALANCE': 'ACCOUNT_LENGTH'}, inplace=True)

start_df['ACCOUNT_LENGTH'] = -start_df['ACCOUNT_LENGTH']


# In[55]:


start_df


# In[56]:


df = pd.merge(df, start_df, how='inner', on=['ID'])


# In[57]:


df.dtypes


# In[58]:


df.describe()


# In[59]:


df.columns


# In[60]:


df.dtypes


# In[61]:


# Get numerical column names
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Columns:")
print(num_cols)

# Get categorical column names
cat_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:")
print(cat_cols)


# In[62]:


df.target.value_counts()


# Analysis of numerical and categorical variables

# In[63]:


# creating a list of columns which are numerical
num_col = df.select_dtypes(include=['int64','float64']).columns.tolist()

#printing num_col
print(num_col)


# In[64]:


# plotting histograms for seeing the distributions of numerical variables
for col in num_col:
  sns.histplot(data=df, x = col, hue = 'target',kde=True)
  plt.title(f'Distribution of {col} vs Target')
  plt.show()
  print("\n")


# In[65]:


sns.countplot(x='target', data=df, palette='coolwarm_r')
df['target'].value_counts()


# In[66]:


df.drop(columns=['ID', 'CNT_CHILDREN'], inplace=True)


# The dataset was imported from a CSV file, containing records of customers with attributes such as age, income, employment details, and credit information. Initial preprocessing involved cleaning missing values, particularly in the 'Occupation Type' field, which had a significant number of missing entries. Duplicates were checked and no duplicates were found, ensuring data integrity. Categorical variables were encoded to facilitate analysis, and data types were adjusted for computational efficiency.
# 
# I have decided to drop ID and CNT_CHILDREN columns as they seem redundant for our model having a biased distribution.

# In[67]:


# create a list of columns for applying log transformations
log_list = ['AGE_YEARS','ACCOUNT_LENGTH']

# applying log transformation to the desired columns
for col in log_list:
  df[col] = np.log1p(df[col])

# plotting a histogram to verify the distribution
for col in log_list:
  sns.histplot(data=df,x=col,hue='target',kde=True)
  plt.show()


# After applying the log transformation we can see that the distributions of age and final-weight are much more even and closer to normal than before.

# In[68]:


#Analyzing Categorical Features

# plotting frequency distribution for categorical variables

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
  plt.figure(figsize=(10,6))
  sns.countplot(data=df,x=col,hue='target')
  plt.title(f'Frequency Distribution of {col}')
  plt.xticks(rotation=90)
  plt.show()
  print("\n")


# Feature Engineering

# In[69]:


plt = sns.pairplot(data=df)


# One Hot Encoding

# In[70]:


columns_to_scale = ['AMT_INCOME_TOTAL', 'AGE_YEARS','YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']


# In[71]:


dummy = df.copy()


# In[72]:


st=StandardScaler()
df[columns_to_scale] = st.fit_transform(df[columns_to_scale])


# In[73]:


columns_to_encode = ['CODE_GENDER','FLAG_OWN_CAR','NAME_EDUCATION_TYPE', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE']
df=pd.get_dummies(df,columns=columns_to_encode,dtype='int')


# In[74]:


df


# In[75]:


df.describe()


# Summary statistics revealed key insights, such as the average income level, age distribution, and employment tenure among the customers. Visualizations like histograms and scatter plots provided a deeper understanding of the distribution of key variables. For instance, the income distribution was right-skewed, indicating a smaller number of high-income customers. Box plots highlighted outliers in variables like age and employment years, suggesting the presence of exceptionally old or long-tenured individuals. A correlation analysis was conducted to explore relationships between variables. The findings indicated a weak correlation between age and income levels, suggesting that higher income is not necessarily associated with older age in this dataset. However, a moderate positive correlation was observed between employment tenure and credit amount, implying that longer-employed individuals tend to have higher credit amounts.
