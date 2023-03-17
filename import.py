# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Evaluate Your Machine Learning Algorithms
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7

# Evaluate using a train and a test set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Evaluate using a train and a test set")
print("Accuracy: %.3f%%" % (result*100.0))

# Evaluate using Cross Validation
num_instances = len(X)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Evaluate using Cross Validation")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# Evaluate using Leave One Out Cross Validation
num_folds = 10
loocv = model_selection.LeaveOneOut()
results = model_selection.cross_val_score(model, X, Y, cv=loocv)
print("Evaluate using Leave One Out Cross Validation")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# Evaluate using Shuffle Split Cross Validation
num_samples = 10
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Evaluate using Shuffle Split Cross Validation")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# Compare Algorithms
h = dataframe.hist()
g = pandas.plotting.scatter_matrix(dataframe)