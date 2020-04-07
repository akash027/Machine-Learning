import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("/home/sky/Documents/1.machine learning/Decision Trees/iris_data.csv")

print(data.head())



data.features = data[["SepalLength", "SepalWidth",  "PetalLength", "PetalWidth"]]
data.targets = data.Class


# with grid search you can find an optimal parameter "parameter tunning

param_grid = {'max_depth': np.arange(1,10)}


# in every iteration data is splitted randomely in cross validation + DecisionTreeClassifier
# initializes the tree randomly: thats why you get diffenrent results

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)


feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.targets, test_size =0.3 )

tree.fit(feature_train, target_train)

tree_pred = tree.predict_proba(feature_test)[:,1]

print("Best parameter with Grid Search: ", tree.best_params_)

