import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import csv file as a dataframe
df = pd.read_csv('.csv')

# Split the dataframe
X = df.drop('target_variable', axis=1).values
y = df['target_variable'].values

# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=, random_state=, stratify=y)

# Instantiate the algorithm
reg = LinearRegression()

# Fit the algorithm to the training set
reg.fit(X_train, y_train)

# Predict on the test set
predictions = reg.predict(X_test)

# Scoring via R² (equivalent to accuracy in classification)
reg.score(X_test, y_test)
