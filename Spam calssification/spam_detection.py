import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn import  svm
from sklearn.model_selection import GridSearchCV


# Load Dataset
data = pd.read_csv("/spam.csv")
print(data.head())


# split in to Trainning and Test data

x = data["EmailText"]
y = data["Label"]


x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Extract Feature
cv = CountVectorizer()
features = cv.fit_transform(x_train)


# build model

## Parameter tunning

tuned_parameters = { 'kernel' : ['linear', 'rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C' : [1,10,100,1000]
    }

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features, y_train)

print(model.best_params_)


# test accuracy
features_test = cv.transform(x_test)

print(model.score(features_test,y_test))



