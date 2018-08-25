# K Nearest Neighbors ('KNN') algorithm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(str(input("Please enter the name of your csv file: "))+'.csv')
print(df.head())
# Procedure: 1) split the data, 2) compute accuracy on test set

# generate feature array and target (or label) array. Except for the target variable, all columns of a df go to the feature variable.
target_variable = str(input("Please define the target variable of your data set: "))
X = df.drop(str(target_variable), axis=1).values
y = df[str(target_variable)].values

# train_test_split function is used to randomly split the data.
# test_size = what proportion of the original data is used for the test set. By default 25% for test, 75% for training
# random_state = generates a number for randomly splitting the data
# stratify = y achieves that the labels in the test and train sets are the same as in the original data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(float(input("Please define the size of the test dataset in percent (without using the % symbol): "))/100), random_state=int(input("Please define the random state for splitting the original dataset: ")), stratify=y)

# n_neighbors = number of neighbors
knn = KNeighborsClassifier(n_neighbors=int(input("Please define the number of nearest neighbors: ")))

# fit the classifier to the training set, i.e. the labelled data with 1) features as a numpy array and 2) targets as a numpy array.
knn.fit(X_train, y_train)

# predict on the test set (unlabeled data) with the '.predict()' method.
prediction = knn.predict(X_test)

# calculate the accuracy with .score() method. Measure the model's performance via its accuracy. Accuracy = number of correct predictions / total number of data points. Compute accuracy on a test set (this must be data which has not been used to train the model).
accuracy = knn.score(X_test, y_test)
print(accuracy)
