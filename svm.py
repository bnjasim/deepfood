import numpy as np
from sklearn import svm

# Read the data into a numpy array
data = np.genfromtxt('Data_SVM.csv', delimiter=",")
data = data[1:] # first row is empty - remove it
train_data = data[:,0:2]
train_labels = data[:,2]

model = svm.SVC(C=1, kernel='poly', degree=3)
model.fit(train_data, train_labels)


print np.sum(model.predict(train_data) == train_labels)/200.0
