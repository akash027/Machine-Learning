import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("/home/sky/Documents/1.machine learning/Decision Trees/iris_data.csv")

print(data.head())



data.features = data[["SepalLength", "SepalWidth",  "PetalLength", "PetalWidth"]]
data.targets = data.Class


feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.targets, test_size =0.3 )


model = DecisionTreeClassifier(criterion = 'entropy')
model.fitted =  model.fit(feature_train, target_train)

model.predictions = model.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

from sklearn.model_selection import cross_val_predict

predicted = cross_val_predict(model, data.features, data.targets, cv=10)

print(accuracy_score(data.targets, predicted))
