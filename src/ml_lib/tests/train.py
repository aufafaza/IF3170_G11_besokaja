from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from ml_lib.model.svm import KernelSVM
from ml_lib.model.dtl import DecisionTree, DecisionTreeC45, DecisionTreeCART

dataset = datasets.load_breast_cancer()
X = np.asarray(dataset.data)
y = 2 * np.asarray(dataset.target) - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 

model = DecisionTreeCART(ccp_alpha=0.01)
model.fit(X_train, y_train)
ax = model.visualize(feature_names=dataset.feature_names, max_depth=None)
plot_path = Path(__file__).resolve().parents[3] / "output" / "decision_tree.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)
ax.figure.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved tree to {plot_path}")
plt.show()


# def accuracy(y_test, y_pred): 
#     return np.sum(y_test == y_pred) / len(y_test) 
#
# # svm.plot_history()
# print(accuracy(y_test, predictions))
