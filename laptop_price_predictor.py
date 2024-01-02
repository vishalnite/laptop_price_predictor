#Reading csv file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('laptop_data.csv')

#Check if there exists a duplicate row
df.duplicated().sum()

#check if there exists an empty value
df.isnull().sum()

#Drop unnamed: 0 column
df.drop(columns = ['Unnamed: 0'], inplace = True)

#Remove GB from Ram and kg from Weight
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')

#Changing type of 'Ram' and 'Weight' from object to int and float respectively
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

import seaborn as sns

#Data is skewed because commodities with high price are tend to be bought less
sns.distplot(df['Price'])

#Number of laptops per brand
df['Company'].value_counts().plot(kind='bar')

#Average price of laptops of each brand
sns.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

#Number of laptops in each type
df['TypeName'].value_counts().plot(kind='bar')

#Average price of laptops of each type
sns.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

#Variation based on the screen size of laptops
sns.distplot(df['Inches'])

#This shows correlation between Inches and Price of laptops
sns.scatterplot(x=df['Inches'], y=df['Price'])

#Type of screen resolution
#This needs to be handled carefully as we need to find info from this column
df['ScreenResolution'].value_counts()

#Creating another column which shows whether a laptop is touchscreen or not
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

#The number of touchscreen laptops are less
df['Touchscreen'].value_counts().plot(kind='bar')

#As expected the touchscreen laptops have a higher price
sns.barplot(x=df['Touchscreen'], y=df['Price'])

#Creating another column which shows whether a laptop has IPS display or not
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

#The number of laptops having IPS display
df['Ips'].value_counts().plot(kind='bar')

#As expected the laptops with IPS display have a higher price
sns.barplot(x=df['Ips'], y=df['Price'])

#Creating a new variable which stores the values split at 'x'
#The first value is X resolution and second value is Y resolution
new = df['ScreenResolution'].str.split('x', n=1, expand=True)

#storing X resolution and Y resolution
#We observe that Y resolution is the desired value but X resolution need to be processed
df['X_res'] = new[0]
df['Y_res'] = new[1]

#Using regular expression to extract the pattern where some digits are available
df['X_res'] = df['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

#Changing type from object to int
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

#Correlation of different columns with Price
df.corr()['Price']

#Creating another column ppi(pixels per inch)
#ppi = sqrt(x^2 + y^2) / Inches
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

#he new column ppi has a strong correlation with price
df.corr()['Price']

#Removing Screen Resolution column as it is not needed anymore
df.drop(columns = ['ScreenResolution'], inplace = True)


# In[41]:


df.head()


# In[42]:


#Also removing Inches, X_res and Y_res as we already have ppi
df.drop(columns = ['Inches', 'X_res', 'Y_res'], inplace = True)


# In[43]:


df.head()


# In[44]:


#Now we need to process 'CPU' column.
#It has a lot of info and it needs to be handled carefully
df['Cpu'].value_counts()


# In[45]:


#Categorizing different CPU types 
#Extracing first three words from this column
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[46]:


df.head()


# In[47]:


#This function categorizes differnt processors under differnt classes
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[48]:


#Creating a new column to store new cpu classes
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[49]:


df.head()


# In[50]:


#Visualizing the newly created column
df['Cpu brand'].value_counts().plot(kind = 'bar')


# In[51]:


#Visualizing cpu brand against price
sns.barplot(x=df['Cpu brand'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


#Removing Cpu and Cpu Name column
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)


# In[53]:


df.head()


# In[54]:


#Analyzing Ram column
df['Ram'].value_counts().plot(kind='bar')


# In[55]:


#Visualizing Ram against price
sns.barplot(x=df['Ram'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[56]:


#Memory column is typical to handle. There are a lot of variations.
df['Memory'].value_counts()


# In[57]:


#Creating 4 differnt columns for HDD, SSD, Flash Storage, Hybrid
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')
new = df['Memory'].str.split('+', n = 1, expand = True)

df['first'] = new[0]
df['first'] = df['first'].str.strip()

df['second'] = new[1]

df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df['second'].fillna('0', inplace = True)

df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid'])
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage'])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD',
                'Layer2Hybrid', 'Layer2Flash_Storage'], inplace=True)


# In[58]:


df.sample(5)


# In[59]:


#Remove Memory column
df.drop(columns=['Memory'], inplace=True)


# In[60]:


df.head()


# In[61]:


#We can observe that Price is negatively correlated with HDD
#But we will keep it because HDD and SSD together give the best result
#Eventually we will drop Hybrid and Flash Storage column because Price is not strongly correlated with them
df.corr()['Price']


# In[62]:


df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)


# In[63]:


df.head()


# In[64]:


#Gpu column has a lot of information but due to less data it does not have much effect
#So we will rely on the brand of Gpu
df['Gpu'].value_counts()


# In[65]:


#Extracting the Gpu brand from Gpu column
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[66]:


df.head()


# In[67]:


df['Gpu brand'].value_counts()


# In[68]:


# Removing the row of the laptop with ARM processor beacause it is only one
df = df[df['Gpu brand'] != 'ARM']


# In[69]:


df['Gpu brand'].value_counts()


# In[70]:


# Variation of price with GPU brand
# Using median as estimator
sns.barplot(x = df['Gpu brand'], y = df['Price'], estimator = np.median)
plt.xticks(rotation = 'vertical')
plt.show()


# In[71]:


# Removing 'Gpu' column as it is not needed anymore
df.drop(columns = ['Gpu'], inplace = True)


# In[72]:


df.head()


# In[73]:


# Exploring Operating systems column
# There are a lot of categories - We need to club them together to form fewer categories
df['OpSys'].value_counts()


# In[74]:


sns.barplot(x = df['OpSys'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[75]:


# Function to categorize operating systems into smaller categories
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[76]:


# Creating a column to store operating system categories
df['os'] = df['OpSys'].apply(cat_os)


# In[77]:


df.head()


# In[78]:


df.drop(columns = ['OpSys'], inplace = True)


# In[79]:


# On an average MacOS has the highest price followed by Windows and Others
sns.barplot(x = df['os'], y = df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[80]:


# Exploring the 'weights' column 
# It is fairly normal but it is bi-modal because we are getting high frequency at two points
sns.distplot(df['Weight'])


# In[81]:


# There is a slight linear relationship between weight and price although it is very weak
# As weight increases there is a slight increase in price
sns.scatterplot(x = df['Weight'], y = df['Price'])


# In[82]:


df.corr()['Price']


# In[83]:


# Exploring the correlation of columns with each other
# The lighest region shows the highest correlation
sns.heatmap(df.corr())


# In[84]:


# The target variable is skewed and it may trouble our machine learnign algorithms
# To convert it into a normal distribution we can use log transformation
sns.distplot(df['Price'])


# In[85]:


# Applying log transformation to target variable
# While extracting x and y variables - we will apply log to y 
# While prediction apply opposite of log i.e, exponent operation
sns.distplot(np.log(df['Price']))


# In[86]:


#Extracting features as X and target variable as y
X = df.drop(columns = ['Price'])
y = np.log(df['Price'])


# In[87]:


X


# In[88]:


y


# In[89]:


# SPlitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 2)


# In[90]:


X_train


# In[91]:


# Categorical data needs to be handled
# We need to convert these columns using OneHotEncoding
# We will build a sklearn pipeline and do the tranformation step in a single line
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error


# In[92]:


# Importing the best algorithms available in sklearn
# We will experiment on all of them as we do not know which algorithm works best for this problem
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ### Linear Regression

# In[104]:


# This is the first step in our pipeline
# We need to give the column numbers excluding 'Price' where OneHotEncoding needs to be applied
# Also we do not want the matrix to be sparse and we do not want to change anything in the numerical columns
step1 = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse = False, drop = 'first'), [0, 1, 7, 10, 11])
], remainder = 'passthrough')

# Second step is just applying the Linear Regression
# This step needs to be changed in order to apply different algorithms
step2 = LinearRegression()

# Creating the pipeline
pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Applying training data to the pipeline
pipe.fit(X_train, y_train)

# Making prediction
y_pred = pipe.predict(X_test)

# Evaluating R2 score and Mean Absolute Error
print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# ### Ridge Regression

# In[94]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha = 10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Lasso Regression

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha = 0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### KNN

# In[96]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors = 3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Decision Tree

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth = 8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### SVM

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel = 'rbf', C = 10000, epsilon = 0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Random Forest

# In[114]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators = 100,
                             random_state = 3,
                             max_samples = 0.5,
                             max_features = 0.75,
                             max_depth = 15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### ExtraTrees

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators = 100,
                           random_state = 3,
                           max_samples = 0.5,
                           max_features = 0.75,
                           max_depth = 15,
                           bootstrap = True)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### AdaBoost

# In[101]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators = 15, learning_rate = 1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Gradient Boost

# In[102]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators = 500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### XGBoost

# In[103]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators = 45, max_depth = 5, learning_rate = 0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ### Voting Regressor

# In[109]:


from sklearn.ensemble import VotingRegressor, StackingRegressor

step1 = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse = False, drop = 'first'), [0, 1, 7, 10, 11])
], remainder = 'passthrough')

rf = RandomForestRegressor(n_estimators = 350, random_state = 3, max_samples = 0.5, max_features = 0.75, max_depth = 15)
gbdt = GradientBoostingRegressor(n_estimators = 100, max_features = 0.5)
xgb = XGBRegressor(n_estimators = 25, learning_rate = 0.3, max_depth = 5)
et = ExtraTreesRegressor(n_estimators = 100, random_state = 3, max_samples = 0.5, max_features = 0.75, max_depth = 10, bootstrap = True)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb', xgb), ('et', et)], weights = [5, 1, 1, 1])

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# ### Stacking

# In[112]:


from sklearn.ensemble import VotingRegressor, StackingRegressor

step1 = ColumnTransformer(transformers = [
    ('col_tnf', OneHotEncoder(sparse = False, drop = 'first'), [0, 1, 7, 10, 11])
], remainder = 'passthrough')

estimators = [
    ('rf', RandomForestRegressor(n_estimators = 350, random_state = 3, max_samples = 0.5, max_features = 0.75, max_depth = 15)),
    ('gbdt', GradientBoostingRegressor(n_estimators = 100, max_features = 0.5)),
    ('xgb', XGBRegressor(n_estimators = 25, learning_rate = 0.3, max_depth = 5))
]

step2 = StackingRegressor(estimators = estimators, final_estimator = Ridge(alpha = 100))

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))


# ### Exporting the Model

# In[116]:


# Exporting Random Forest because best result was obtained there
# Run Random Forest cell before running this cell
import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))


# In[115]:


df


# In[117]:


X_train


# In[ ]:




