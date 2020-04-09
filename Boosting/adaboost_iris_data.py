import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("/iris_data.csv")

print(data.head())


data.features = data[["SepalLength" , "SepalWidth" , "PetalLength" , "PetalWidth" ]]
data.target = data.Class


feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.target, test_size=0.3)


model = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=123)

model.fitted = model.fit(feature_train, target_train)

model.predictions = model.predict(feature_test)


print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

