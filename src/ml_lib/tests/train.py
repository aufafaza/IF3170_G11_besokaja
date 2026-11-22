from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from ml_lib.model.svm import LinearSVM
X, y = datasets.load_breast_cancer(return_X_y=True)
X = np.asarray(X)
y = 2 * np.asarray(y) - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = LinearSVM(n_features=X_train.shape[1])

svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

def accuracy(y_test, y_pred): 
    return np.sum(y_test == y_pred) / len(y_test) 

svm.plot_history()
print(accuracy(y_test, predictions))
