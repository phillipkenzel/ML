# simple, interactive K Nearest Neighbors ('KNN') model template.
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Asks the user to specify the name of a CSV file.
df = pd.read_csv(str(input("Please enter the name of your CSV file: "))+'.csv')
print(df.head())

# Asks the user to define the target (or label) variable.
target_variable = str(input("Please define the target variable of your data set: "))
X = df.drop(str(target_variable), axis=1).values
y = df[str(target_variable)].values

# Asks the user to define test_size and random_state hyperparameters.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(float(input("Please define the size of the test dataset in percent (without using the % symbol): "))/100), random_state=int(input("Please define the random state for splitting the original dataset: ")), stratify=y)

# Asks the user to define number of nearest neighbors.
knn = KNeighborsClassifier(n_neighbors=int(input("Please define the number of nearest neighbors: ")))

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

accuracy = knn.score(X_test, y_test)
print(accuracy)
