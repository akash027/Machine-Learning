import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score


#the digit dataset
digits = datasets.load_digits()

print(digits)

images_and_labels = list(zip(digits.images, digits.target))

print(images_and_labels)

for index, (image, label) in enumerate(images_and_labels[:6]):
    plt.subplot(2, 3, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i '% label)


# to apply a classifier on this data, we need to flatten the image, 
# to turn data in a (sample, feature) matrix:

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

print(data.shape)


classifier = svm.SVC(gamma=0.001)

# we learn the digits on the first half of the digits
trainTestSplit = int(n_samples * 0.75)
classifier.fit(data[:trainTestSplit], digits.target[:trainTestSplit])

    
# Predict the value of the digits on second half
expected = digits.target[trainTestSplit:]
predicted = classifier.predict(data[trainTestSplit:])

print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))


#lets test on the last few images
plt.imshow(digits.images[-3],cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for image:", classifier.predict(data[-3].reshape(1,-1)))