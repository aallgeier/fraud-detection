"""
Descriptoon: Re-did parts of original project
Author: Allegra Allgeier
COLAB link: https://colab.research.google.com/drive/17T-_oMcmu-0jNXn0ljvkhn9oPZaMKvzJ?usp=sharing
"""

import pandas as pd
import numpy as np
import sklearn 
from sklearn.decomposition import PCA

df_trans = pd.read_csv('ieee-fraud-detection/train_transaction.csv')  
df_id = pd.read_csv('ieee-fraud-detection/train_identity.csv')

print(df_trans)
print(df_id)

# Merge transaction and identity
## Inner merge
df_merged = df_trans.merge(df_id)
print(df_trans)

y = df_merged["isFraud"]

"""
Dropping data
"""
# Remove features we won't use effectively
X = df_merged.drop(['TransactionID',"isFraud", "TransactionDT"], axis=1)
print(X)

# Handle NaN values (Remove columns with more than 90% NaN)
X.dropna(axis='columns')
X = X.loc[:, X.isnull().mean() < .9]
print(X)

# Extract numerical features and fill numerical NaNs with feature means.
numeric_data = X.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())
print(numeric_data)
#categorical_data = X.select_dtypes(exclude=[np.number])
# categorical_data
# categorical_data = pd.factorize(categorical_data)
# print(categorical_data)

"""
Dimensionality reduction
"""
# Remove low variance features as it does not contribute to the fitting if it's basically constant.
normalized_df=(numeric_data-numeric_data.min())/(numeric_data.max()-numeric_data.min())
numeric_data = numeric_data.loc[:, normalized_df.std() > .05]
print(numeric_data)

# Remove features correlated to other features
# Code from https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# Create correlation matrix
corr_matrix = numeric_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
numeric_data.drop(to_drop, axis=1, inplace=True)

print(numeric_data)

"""
Dimensionality reduction - PCA
"""
# Standardize data (mean 0 SD 1)
scalar = sklearn.preprocessing.StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(numeric_data)) 
print(scaled_data)

pca = PCA(n_components=25)
reduced_data = pca.fit_transform(scaled_data)
print(reduced_data.shape)