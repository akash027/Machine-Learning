import numpy as np
from keras.models  import  Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

data = load_iris()

features = data.data
y = data.target.reshape(-1,1)


encoder = OneHotEncoder()
targets = encoder.fit_transform(y)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)


model = Sequential()

#first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))


#we can define loss functoin MSE or negative log likelihood
#optimizer will find the right adjustments for the weights 

optimizer = Adam(lr=0.005)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, batch_size=20, verbose=2)

results = model.evaluate(test_features, test_targets)

print("Accuracy on the test dataset: %.2f"% results[1])
