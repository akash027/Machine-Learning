import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets


dataset = datasets.load_iris()

print(dataset)

features = dataset.data
target = dataset.target

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features,target, test_size=0.30)


#you can set parameters as well
#model = svm.SVC(gamma=0.001, C=100)

model = svm.SVC()
fittedModel = model.fit(featureTrain,targetTrain)

predictions = fittedModel.predict(featureTest)


print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))  