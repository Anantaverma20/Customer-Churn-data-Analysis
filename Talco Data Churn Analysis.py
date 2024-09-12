#!/usr/bin/env python
# coding: utf-8

# # Problems to be solved
# #### 1. Customers from which city have high churn rate?
# #### 2. Which offer reduces the churn rate most.
# #### 3. What are the major reasons for the customer churn and how can we reduce it.

# # Information on Dataset

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format


# In[3]:


data = pd.read_csv("E:/My projects/telco.csv")


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isna().sum()


# In[8]:


data.nunique()


# In[30]:


import pandas as pd

# Calculate correlations
correlations = data.corr()

# Show correlation with the target variable 'Churn'
print(correlations['Churn Label'].sort_values(ascending=False))


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert 'Churn Label' column to numeric
data['Churn Label'] = data['Churn Label'].map({'Yes': 1, 'No': 0})

# Verify conversion
print(data['Churn Label'].unique())  # Should output: array([1, 0])

# Sort churn rates by city
churn_rate_by_city = data.groupby('City')['Churn Label'].mean().sort_values(ascending=False)
print("Churn rate by city:")
print(churn_rate_by_city)

# Get the top 10 cities with the highest churn rate
top_10_cities = churn_rate_by_city.head(10)

# Plot for Top 10 Cities
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_cities.index, y=top_10_cities.values, palette='viridis', alpha=0.6)
plt.xticks(rotation=45)
plt.title('Top 10 Cities with Highest Churn Rate')
plt.ylabel('Churn Rate')
plt.xlabel('City')
plt.show()

# 2. Which offer reduces the churn rate most?
churn_rate_by_offer = data.groupby('Offer')['Churn Label'].mean().sort_values(ascending=False)
print("Churn rate by offer:")
print(churn_rate_by_offer)

plt.figure(figsize=(10, 6))
sns.barplot(x=churn_rate_by_offer.index, y=churn_rate_by_offer.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Churn Rate by Offer')
plt.ylabel('Churn Rate')
plt.xlabel('Offer')
plt.show()

# 3. What are the major reasons for customer churn?
churn_reason_counts = data['Churn Reason'].value_counts()
print("Top churn reasons:")
print(churn_reason_counts)

plt.figure(figsize=(10, 6))
sns.barplot(y=churn_reason_counts.index, x=churn_reason_counts.values, palette='magma')
plt.title('Top Reasons for Customer Churn')
plt.xlabel('Number of Customers')
plt.ylabel('Churn Reason')
plt.show()


# #### *Insights*
# 1. The following cities have been identified as having the highest customer churn rates:
# Eldridge          
# Smith River       
# Twain             
# Johannesburg      
# Riverbank <br>
# Wrightwood <br>
# Boulder Creek <br>
# South Lake Tahoe <br>
# San Dimas <br>
# Acampo <br>
# 2.Two specific promotional offers have been found to significantly reduce the churn rate:
# 
# * Offer A: Reduces churn by 93%.
# * Offer B: Reduces churn by 88%.
# 
# 3. The primary reason customers are leaving is due to the perception that competitors provide better devices. Secondary reasons include:
# * Competitor made a better offer.
# * Negative customer support experience, specifically related to the attitude of support personnel.

# In[10]:


''' Since some of these columns have only 1 unique value and 
some of them are not required for this analysis 
so we are removing them from the dataset '''

columns_to_drop = ['Country', 'State', 'Quarter', 'Zip Code', 'Latitude', 'Longitude','City']
existing_columns = data.columns.intersection(columns_to_drop)
data = data.drop(existing_columns, axis=1)


# In[11]:


# Compute the correlation matrix
corr_matrix = data.corr()

# Print correlations with the target variable 'Churn Label'
print(corr_matrix['Churn Label'].sort_values(ascending=False))

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# #### *Insights*
# 1. Churn label is highly negatively correlated to satisfaction score which makes sense as if the customer is satisfied then he won't leave the company.
# 2. Churn score and churn label are positively correlated.
# 

# # Data Preparation

# In[12]:


# replacing the null values in the two columns with "not provided/Unknown"

data['Churn Category'].fillna('Unknown', inplace=True)
data['Churn Reason'].fillna('Not Provided', inplace=True)


# In[13]:


data.isna().sum().sum()


# In[14]:


#code taken from github
features_cat = list(data.select_dtypes(exclude = ['int64','float64']))
features_num = list(data.select_dtypes(include = ['int64','float64']))


# In[32]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Convert categorical variables to numeric using LabelEncoder for correlation analysis
encoder = LabelEncoder()
data_encoded = data.copy()
for column in features_cat:
    data_encoded[column] = encoder.fit_transform(data[column])

# Calculate correlations again
correlations = data_encoded.corr()
print(correlations['Churn Label'].sort_values(ascending=False))


# In[15]:


features_cat.remove('Customer ID')


# In[16]:


data.drop('Customer ID', axis = 1, inplace=True)


# In[17]:


# now are going to use ordinarEncoding for columns with cardinatity <= 2
# and for more than that we are going to use OneHotEncoding

feature_series = data[features_cat].nunique().sort_values(ascending=False)
feature_series


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

# Step 1: Identify categorical features based on cardinality
categorical_features = data.select_dtypes(include=['object']).columns
onehot_features = [col for col in categorical_features if data[col].nunique() > 2 and col != 'Churn Label']
label_features = [col for col in categorical_features if data[col].nunique() == 2 and col != 'Churn Label']

# Step 2: Split the data into training and test sets
X = data.drop(columns=['Churn Label'])  # Features
y = data['Churn Label']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Apply One-Hot Encoding to columns with more than 2 unique values
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder on the training data only
encoded_train_array = onehot_encoder.fit_transform(X_train[onehot_features])

# Transform the test data using the fitted encoder
encoded_test_array = onehot_encoder.transform(X_test[onehot_features])

# Get the new column names for the one-hot encoded features
encoded_columns = onehot_encoder.get_feature_names_out(onehot_features)

# Convert the encoded arrays to DataFrames
encoded_train_df = pd.DataFrame(encoded_train_array, columns=encoded_columns, index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test_array, columns=encoded_columns, index=X_test.index)

# Step 4: Apply Label/Ordinal Encoding to columns with exactly 2 unique values
label_encoders = {}
for col in label_features:
    if col in X_train.columns:  # Ensure the column is in the DataFrame
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le  # Store the label encoder for future use if needed

# Step 5: Drop the original categorical features that were one-hot encoded
X_train = X_train.drop(onehot_features, axis=1)
X_test = X_test.drop(onehot_features, axis=1)

# Join the encoded DataFrames with the original DataFrames
X_train = X_train.join(encoded_train_df)
X_test = X_test.join(encoded_test_df)

# Output the transformed training and test sets
print("Transformed Training Set:")
print(X_train.head(2).T)

print("\nTransformed Test Set:")
print(X_test.head(2).T)


# In[22]:


from matplotlib.ticker import FuncFormatter
# DataFrame `train_data` that includes both X_train and y_train
train_data = pd.concat([X_train, y_train], axis=1)

# Initialize lists to store column names
one_hot_encoded_cols = []
label_encoded_cols = []

# Loop through all columns in the DataFrame
for col in train_data.columns:
    if train_data[col].dtype == 'int64' or train_data[col].dtype == 'float64':
        unique_values = train_data[col].nunique()
        
        # Identify one-hot encoded columns (binary columns with only two unique values)
        if unique_values == 2 and set(train_data[col].unique()) <= {0, 1}:
            one_hot_encoded_cols.append(col)
        # Identify label encoded columns (categorical columns with a limited number of unique integer values)
        elif unique_values < 20:  # Adjust threshold based on your understanding of the data
            label_encoded_cols.append(col)

# Now plot the bivariate analysis using identified columns

# Function to format the x-axis labels
def custom_format(x, pos):
    if x == 0 or x == 1:
        return f'{int(x)}'  # Remove decimal for 0 and 1
    else:
        return f'{x:.2f}'  # Format other numbers to 2 decimal places

    
# Plot for Label Encoded Columns
for col in label_encoded_cols:
    if col != 'Churn Label':  # Skip target column
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=train_data, x=col, hue='Churn Label')
        ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge')
        plt.title(f'Count Plot: {col} vs. Churn Label')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

# Plot for One-Hot Encoded Columns
for col in one_hot_encoded_cols:
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=train_data, x=col, hue='Churn Label')
    ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')
    plt.title(f'Count Plot: {col} vs. Churn Label')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()


# In[23]:


from matplotlib.ticker import FuncFormatter
# DataFrame `train_data` that includes both X_train and y_train
test_data = pd.concat([X_test, y_test], axis=1)

# Initialize lists to store column names
one_hot_encoded_cols = []
label_encoded_cols = []

# Loop through all columns in the DataFrame
for col in test_data.columns:
    if test_data[col].dtype == 'int64' or test_data[col].dtype == 'float64':
        unique_values = test_data[col].nunique()
        
        # Identify one-hot encoded columns (binary columns with only two unique values)
        if unique_values == 2 and set(test_data[col].unique()) <= {0, 1}:
            one_hot_encoded_cols.append(col)
        # Identify label encoded columns (categorical columns with a limited number of unique integer values)
        elif unique_values < 20:  # Adjust threshold based on your understanding of the data
            label_encoded_cols.append(col)

# Now plot the bivariate analysis using identified columns

# Function to format the x-axis labels
def custom_format(x, pos):
    if x == 0 or x == 1:
        return f'{int(x)}'  # Remove decimal for 0 and 1
    else:
        return f'{x:.2f}'  # Format other numbers to 2 decimal places

    
# Plot for Label Encoded Columns
for col in label_encoded_cols:
    if col != 'Churn Label':  # Skip target column
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=test_data, x=col, hue='Churn Label')
        ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge')
        plt.title(f'Count Plot: {col} vs. Churn Label')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

# Plot for One-Hot Encoded Columns
for col in one_hot_encoded_cols:
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=test_data, x=col, hue='Churn Label')
    ax.xaxis.set_major_formatter(FuncFormatter(custom_format))
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')
    plt.title(f'Count Plot: {col} vs. Churn Label')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()


# # Model Creation

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=10,              # Maximum depth of each tree
    min_samples_split=10,      # Minimum number of samples required to split an internal node
    min_samples_leaf=5,        # Minimum number of samples required to be at a leaf node
    random_state=42            # Ensure reproducibility
)

# Step 2: Fit the model on the training data
rf_model.fit(X_train, y_train)

# Step 3: Predict on the test data
y_test_pred_rf = rf_model.predict(X_test)

# Step 4: Evaluate the model
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print("Random Forest Test Classification Report:")
print(classification_report(y_test, y_test_pred_rf))

# Optional: Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)
print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)


# In[27]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Convert target labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on the training target and transform both training and test targets
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=100,          # Number of trees
    max_depth=4,               # Maximum depth of each tree
    learning_rate=0.1,         # Step size shrinkage used to prevent overfitting
    subsample=0.8,             # Subsample ratio of the training instances
    colsample_bytree=0.8,      # Subsample ratio of columns when constructing each tree
    random_state=42,           # Ensure reproducibility
    use_label_encoder=False,   # To avoid label encoder warning in recent versions
    eval_metric='logloss'      # Evaluation metric
)

# Fit the model on the training data
xgb_model.fit(X_train, y_train_encoded)

# Predict on the test data
y_test_pred_xgb = xgb_model.predict(X_test)

# Decode the predicted labels to original label names ('No', 'Yes')
y_test_pred_xgb_decoded = label_encoder.inverse_transform(y_test_pred_xgb)

# Evaluate the model
print("XGBoost Test Accuracy:", accuracy_score(y_test, y_test_pred_xgb_decoded))
print("XGBoost Test Classification Report:")
print(classification_report(y_test, y_test_pred_xgb_decoded))

# Optional: Confusion Matrix
conf_matrix_xgb = confusion_matrix(y_test, y_test_pred_xgb_decoded)
print("XGBoost Confusion Matrix:")
print(conf_matrix_xgb)


# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# Feature importance for Random Forest
rf_feature_importance = rf_model.feature_importances_
features = X_train.columns
importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': rf_feature_importance})
importance_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df_rf['Feature'], importance_df_rf['Importance'])
plt.xticks(rotation=90)
plt.title('Random Forest Feature Importance')
plt.show()

# Feature importance for XGBoost
xgb_feature_importance = xgb_model.feature_importances_
importance_df_xgb = pd.DataFrame({'Feature': features, 'Importance': xgb_feature_importance})
importance_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df_xgb['Feature'], importance_df_xgb['Importance'])
plt.xticks(rotation=90)
plt.title('XGBoost Feature Importance')
plt.show()


# # Summary 
# 
# 1. The following cities have been identified as having the highest customer churn rates:
# Eldridge          
# Smith River       
# Twain             
# Johannesburg      
# Riverbank <br>
# Wrightwood <br>
# Boulder Creek <br>
# South Lake Tahoe <br>
# San Dimas <br>
# Acampo <br>
# 
# Actionable Insight: These cities should be prioritized for targeted customer retention strategies, such as personalized offers, enhanced customer service, and proactive engagement to prevent further churn.
#   
# 2. Two specific promotional offers have been found to significantly reduce the churn rate:
# 
# * Offer A: Reduces churn by 93%.
# * Offer B: Reduces churn by 88%. 
# <br>
# 
# Recommendation: These offers should be expanded and targeted at at-risk customers, particularly in regions with high churn rates, to further decrease customer attrition. It may also be beneficial to explore variations of these offers to maximize customer retention across different segments.
# 
# 3. The primary reason customers are leaving is due to the perception that competitors provide better devices. Secondary reasons include:
# * Competitor made a better offer.
# * Negative customer support experience, specifically related to the attitude of support personnel.
# <br>
# 
# Recommendation: To address these issues, the company should consider:
#    * Improving device offerings to remain competitive in the market.
#    * Revisiting pricing and offer structures to ensure they are competitive with rival firms.
#    * Enhancing customer support training, with a focus on improving the customer experience and addressing the support team's attitude and responsiveness.
# 
# ##### Lastly the feature importance graphs from both the models imply that churn category and churn reason are the two columns that is leading to data leakage and that is the reason why we are getting a 100% accuracy on test as well as train data which means overfitting o neither one of our models.

# ---
