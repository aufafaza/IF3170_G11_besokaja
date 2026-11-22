from sklearn import datasets 
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from ml_lib.model.svm import KernelSVM
from ml_lib.model.dtl import DecisionTree, DecisionTreeC45, DecisionTreeCART

X, y = datasets.load_breast_cancer(return_X_y=True)
X = np.asarray(X)
y = 2 * np.asarray(y) - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 

search = GridSearchCV(
    DecisionTreeCART(),
    {"ccp_alpha": [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]},
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print("Best alpha:", search.best_params_["ccp_alpha"])
print("Validation accuracy:", search.best_score_)
model = search.best_estimator_

# def accuracy(y_test, y_pred): 
#     return np.sum(y_test == y_pred) / len(y_test) 
#
# # svm.plot_history()
# print(accuracy(y_test, predictions))
