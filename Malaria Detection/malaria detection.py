import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib


# load dataset

data = pd.read_csv("/home/sky/Documents/1.machine learning/Malaria Detection/dataset.csv")

print(data.head())



#split data into training and test data

x = data.drop(["Label"], axis=1)
y = data["Label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# drop na values

x_train = x_train.fillna(0)
x_test = x_test.fillna(0)

#Build model

model = RandomForestClassifier(n_estimators=100, max_depth=5)

model.fit(x_train,y_train)


# making predictions

predictions = model.predict(x_test)

print(metrics.classification_report(predictions, y_test))


