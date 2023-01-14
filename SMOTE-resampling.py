"""
Descriptoon: Re-did parts of original project
Author: Allegra Allgeier
COLAB link: https://colab.research.google.com/drive/1ntFy_XzuGmaF7nsG0siByoWxnzCH1W0J?usp=sharing
"""

import pandas as pd
from imblearn.over_sampling import SMOTE

X = pd.read_csv("preprocessed_X")
y = pd.read_csv('preprocessed_y')["isFraud"]
X = X.drop(['Unnamed: 0'], axis=1)

print("Num datapoints:", len(y))
print("Num fradulent datapoints:", sum(y))
print("Percentage of fraud:", 100*sum(y)/len(y))

#  The number of samples in the different classes will be equalized.
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X, y)
print("(resampled) Num fradulent datapoints:", sum(y_resampled))
print("(resampled) Percentage of fraud:", 100*sum(y_resampled)/len(y_resampled))
