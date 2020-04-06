import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


data = pd.read_csv("/home/sky/Documents/1.machine learning/K-Nearest neighbour/credit_data.csv")

print(data.head())

data.features = data[["income","age","loan"]]
data.target = data.default

data.features = preprocessing.MinMaxScaler().fit_transform(data.features)

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.target, test_size=0.3)


model = KNeighborsClassifier(n_neighbors=20) # k value

fit_model = model.fit(feature_train,target_train)

predictions = fit_model.predict(feature_test)


print(confusion_matrix(target_test,predictions))
print(accuracy_score(target_test,predictions))



# finding best k value

cross_valid_scores = []

for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, data.features, data.target, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())

print("Optimal k with cross-validation", np.argmax(cross_valid_scores))
