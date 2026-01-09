from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X,y = mnist["data"],mnist["target"]
y = y.astype(np.int8)

X_train, X_test, y_train , y_test = X[:60000], X[60000:], y[:60000], y[60000:]
index = int(len(X)/10)
digit = X[index]
digit_shape = digit.reshape(1,-1)

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index] , y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
prediction = sgd_clf.predict(digit_shape)
print(prediction)


#iterate through all digits and run classifier on each one to identify the digit and print the digit as well