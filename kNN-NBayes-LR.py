import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import time

## Scoring
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load resampled training data
train_x = pd.read_csv("data/smote_x_train.csv")
train_y = pd.read_csv("data/smote_y_train.csv")

# Change label datatype from float to int otherwsie gives error
train_y = train_y.astype('int')

print(train_x)
print(train_y)

# Load testing data
test_x = pd.read_csv("data/X_test.csv")
test_y = pd.read_csv("data/y_test.csv")

# Change label datatype from float to int otherwsie gives error
test_y = test_y.astype('int')
print(test_y)
test_y = np.squeeze(test_y.to_numpy())

print(test_x)

# Calculate scores
def get_scores(true_label, preds):
  accuracy = accuracy_score(true_label, preds)
  precision = precision_score(true_label, preds)
  recall = recall_score(true_label, preds)
  f1= f1_score(true_label, preds)

  return accuracy, precision, recall, f1

## Naive Bayes ## 
## Continuous features (model with gaussian distribution) ##
gnb_begin = time.time()
gnb = GaussianNB().fit(train_x, train_y)
print()
gnb_preds = gnb.predict(test_x)
print("Gaussian Naive Bayes")
print("Train + test time (s):", time.time() - gnb_begin)
accuracy_nb, precision_nb, recall_nb, f1_score_nb = get_scores(test_y, gnb_preds)
print("Accuracy:", accuracy_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("f1_score:", f1_score_nb)

## Logistic Regression ## 
lg_begin = time.time()
clf = LogisticRegression(random_state=0).fit(train_x, train_y)
print()
LR_pred = clf.predict(test_x)
print("Logistic Regression")
print("Train + test time (s):", time.time() - lg_begin)
accuracy_lg, precision_lg, recall_lg, f1_score_lg = get_scores(test_y, LR_pred)

print("Accuracy:", accuracy_lg)
print("Precision:", precision_lg)
print("Recall:", recall_lg)
print("f1_score:", f1_score_lg)

## KNN ## 
# Choose odd number for neighbors
num_neighbors = [1, 3, 5, 7, 11, 21, 35, 51, 101, 1001, 5001]
accuracies_knn = []
precisions_knn = []
recalls_knn =    []
f1_scores_knn =  []

for k in num_neighbors:
  knn_begin = time.time()
  nbrs = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
  nbrs.fit(train_x, train_y)
  print()
  print("KNN: " +str(k)+" neighbors")
  knn_pred = nbrs.predict(test_x) 
  print("Train + prediction time (s)　:", time.time() - knn_begin)
  accuracy_knn, precision_knn, recall_knn, f1_score_knn = get_scores(test_y, knn_pred)
  print("Accuracy:", accuracy_knn)
  print("Precision:", precision_knn)
  print("Recall:", recall_knn)
  print("f1_score:", f1_score_knn)

  accuracies_knn.append(accuracy_knn)
  precisions_knn.append(precision_knn)
  recalls_knn.append(recall_knn)


#### FIRST PLOT ####
plt.figure(figsize=(5,4))
plt.xlabel("Number of neighbors")
plt.ylabel("score")
plt.xlim(-100, 1200)

## Accuracies ##
plt.scatter(num_neighbors, accuracies_knn, label = "accuracy", marker = "^", color = "red")
plt.annotate("k="+str(num_neighbors[0]), (num_neighbors[0], accuracies_knn[0]))
#plt.annotate("k="+str(num_neighbors[-2]), (num_neighbors[-2], accuracies_knn[-2]))
plt.plot(num_neighbors, accuracies_knn, color = "red")

## Precision ## 
plt.scatter(num_neighbors, precisions_knn, label = "precision", marker = "*")
plt.annotate("k="+str(num_neighbors[0]), (num_neighbors[0], precisions_knn[0]))
plt.plot(num_neighbors, precisions_knn)

## f1_scores ##
plt.scatter(num_neighbors, f1_scores_knn, label = "f1-score", marker = "s")
plt.annotate("k="+str(num_neighbors[-2]), (num_neighbors[-2], f1_scores_knn[-2]))
plt.plot(num_neighbors, f1_scores_knn)

plt.legend()
plt.show()

#### SECOND PLOT ####
## Recall ##
plt.figure(figsize=(5,4))
plt.xlabel("Number of neighbors")
plt.ylabel("score")
plt.xlim(-100, 1200)
plt.scatter(num_neighbors, recalls_knn, label = "recall", color = "green")
plt.annotate("k="+str(num_neighbors[0]), (num_neighbors[0],  recalls_knn[0]))
#plt.annotate("k="+str(num_neighbors[-2]), (num_neighbors[-2],  recalls_knn[-2]))
plt.plot(num_neighbors, recalls_knn, color = "green")
plt.legend()
plt.show()