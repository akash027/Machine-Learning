import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


data = pd.read_csv("/home/sky/Documents/1.machine learning/Boosting/wine.csv", sep=";")

print(data.head())

print(data.columns)


def isTasty(quality):
    if quality >=7:
        return 1
    else:
        return 0
    
print(data['quality'].value_counts())


features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']]
data['tasty'] = data['quality'].apply(isTasty)
targets = data["tasty"]



feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

#parameter tunning

param_dist = {
            'n_estimators': [50,10,200],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
            }


grid_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_dist, cv=10)
grid_search.fit(feature_train, target_train)


print("Optimal parameters :",grid_search.best_params_)

preds = grid_search.predict(feature_test)


print(confusion_matrix(target_test, preds))
print(accuracy_score(target_test, preds))
