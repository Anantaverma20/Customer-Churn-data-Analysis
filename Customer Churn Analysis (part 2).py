#!/usr/bin/env python
# coding: utf-8

# # Modeling

# In[1]:


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


# In[2]:


data = pd.read_csv("E:/My projects/telco.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isna().sum()


# In[7]:


data.nunique()


# In[24]:


''' Since some of these columns have only 1 unique value and 
some of them are not required for this analysis 
so we are removing them from the dataset '''

columns_to_drop = ['Country', 'State', 'Quarter', 'Zip Code', 'Latitude', 'Longitude','City', 'Churn Category', 'Churn Reason', 'Churn Score', 'Customer ID', 'Customer Status', 'CLTV']
existing_columns = data.columns.intersection(columns_to_drop)
data = data.drop(existing_columns, axis=1)


# In[25]:


data.isnull().sum().sum()


# In[26]:


#code taken from github
features_cat = list(data.select_dtypes(exclude = ['int64','float64']))
features_num = list(data.select_dtypes(include = ['int64','float64']))


# In[27]:


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


# In[28]:


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


# In[29]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create the model
rf_model = RandomForestClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean()}')


# In[31]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
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


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming you have already trained your model and have the predicted values
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)  # or whichever model you are using


# Fit the model with training data

# Convert y_test and y_pred to NumPy arrays (for better compatibility)
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

# Create a scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_np)), y_test_np, color='blue', label='Actual')
plt.scatter(range(len(y_pred_np)), y_pred_np, color='red', alpha=0.5, label='Predicted')

plt.title('Actual vs Predicted Values (Test Data)')
plt.xlabel('Index')
plt.ylabel('Churn Label')
plt.legend()
plt.show()


# In[40]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ####  Summary
# Cross validation score and xgb accuracy comes our to be 96% which is a high score and therefore this model is good for predicting the outcome i.e. churn label when the inputs are given to the model XGboost model. 

# ---
