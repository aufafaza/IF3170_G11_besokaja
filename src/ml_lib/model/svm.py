import cvxopt
import matplotlib.pyplot as plt
import numpy as np

from ml_lib.model.base import BaseModel


class LinearSVM(BaseModel):
    '''
        This class implements Soft Margin SVM

        Which uses the cost function as followw:
            min(w, b)[1/2 mag(w)^2 + C * {sum of max (0, 1 - y_i (wx_i + b)}]
            equation at rhs of plus sign is hinge loss function
            equation at lhs of plus sign is the minimization of w
    '''

    def __init__(self, C=1, kernel='linear', degree=3, n_iters=1000, learning_rate=0.001):
        # we can seed here
        self.w = None
        self.b = None
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.n_iters=n_iters
        self.learning_rate = learning_rate
        self.history = {}


    def predict(self, X) -> np.ndarray:
        if self.w is None or self.b is None:
            raise RuntimeError("LinearSVM must be fitted before calling predict")
        X = np.asarray(X, dtype=float)
        m = X.shape[0]

        y_preds = np.zeros(m)
        # these will use the final gradient descended w and b resulting from fit.
        for i in range(m):
            y_preds[i] = np.dot(self.w, X[i]) + self.b
        y_preds = np.sign(y_preds)

        return y_preds

    def fit(self, X, y) -> BaseModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand()
        self.history, new_w, new_b = self._gradient_descent(X, y, self.w, self.b)

        # update to the final w and b
        self.w, self.b = new_w, new_b

        return self

    def calc_hinge_loss(self, X_i, y_i, w, b):
        return max(0, 1 - y_i * (np.dot(w, X_i) + b))

    def calc_cost(self, X, y, w, b, C=1):
        m = X.shape[0]
        margin_cost = np.dot(w, w) / 2

        sum = 0
        for i in range(m):
            sum += self.calc_hinge_loss(X[i], y[i], w, b)


        sum *= C

        return margin_cost + sum

    def _calc_gradient(self, X, y, w, b):
        m = X.shape[0]

        dj_dw = w.copy()
        dj_db = 0

        for i in range(m):
            hinge_condition = (1 - y[i] * (np.dot(w, X[i]) + b)) > 0

            if hinge_condition:
                dj_dw -= self.C * y[i] * X[i]
                dj_db += -self.C * y[i]

        return dj_dw, dj_db

    def _gradient_descent(self, X, y, w, b):
        history = {0: self.calc_cost(X, y, w, b, self.C)}
        print(f"Iter\t0\tCost\t{history[0]}")

        for iter in range(1, self.n_iters + 1):
            dj_dw, dj_db = self._calc_gradient(X, y, w, b)
            w = w - self.learning_rate * dj_dw
            b = b - self.learning_rate * dj_db

            if iter % 100 == 0 or iter == self.n_iters:
                cost = self.calc_cost(X, y, w, b, self.C)
                print(f"Iter\t{iter}\tCost\t{cost}")
                history[iter] = cost

        return history, w, b

    # debug
    def svm_plot(self, X, y):
        xlim = np.array([X[:, 0].min()-1, X[:, 0].max()+1])
        ylim = np.array([X[:, 1].min()-1, X[:, 1].max()+1])

        # original_y_plot
        plt.figure(figsize=(4,3))
        plt.scatter(X[:, 0], X[:, 1], c=y)

        plt.title("y_true")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        # y_preds
        y_preds = self.predict(X)
        plt.figure(figsize=(4,3))
        plt.scatter(X[:, 0], X[:, 1], c=y_preds)

        plt.title("y_preds")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])


        # decision boundary plot
        xx = xlim
        yyn = (-1 -self.b - (self.w[0] * xx)) / self.w[1]  # <= -1 eq
        yym = (-self.b - (self.w[0] * xx)) / self.w[1] # == 0 eq
        yyp = (1 -self.b - (self.w[0] * xx)) / self.w[1] # >= 1 eq

        plt.plot(xx, yyn, color='red', linestyle='dotted')
        plt.plot(xx, yym, color='black')
        plt.plot(xx, yyp, color='blue', linestyle='dotted')

        # hinge loss plot
        n_examples = 50
        t = np.zeros(n_examples)
        loss = np.zeros(n_examples)

        for i in range(n_examples):
            t[i] = y[i] * (np.dot(self.w, X[i]) + self.b)
            loss[i] = self.calc_hinge_loss(X[i], y[i], self.w, self.b)
        plt.figure(figsize=(4,3))
        plt.scatter(t, loss)
        plt.title("loss")


        plt.show()

    def plot_history(self, show=True):
        if not self.history:
            raise RuntimeError("LinearSVM must be fitted before plotting training history")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(self.history), list(self.history.values()), marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("SVM gradient descent history")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()
        return ax


class KernelSVM(BaseModel):
    '''
    SVM using kernels, i.e. rbf (gaussian) and polynomial

    class is seperated due to difference in mathematical computation
    '''
    def __init__(self, C=1, kernel='rbf', gamma=None, degree=3, n_iters=1000, learning_rate=0.001, coef=None):
        # we can seed here

        # self.b = np.random.rand()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.n_iters=n_iters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.coef = coef
        # self.history = {}
        self.a: np.ndarray | None = None
        self.support_vectors: np.ndarray | None = None
        self.support_vector_labels: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, X, y) -> BaseModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # samples and features
        m, n = np.shape(X)

        self.gamma_ = 1 / n if self.gamma is None else self.gamma

        # we want to initialize the equation of the kernel
        kernel_function = self._init_kernel(
            degree = self.degree,
            gamma = self.gamma_,
            coef = self.coef
        )

        # create the kernel matrix
        kernel_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                kernel_matrix[i, j] = kernel_function(X[i], X[j])

        # quadratic optimization
        P, q, A, b, G, h = self._init_quadratic_matrix(y, m, kernel_matrix)

        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        if (
            minimization['status'] != 'optimal'
            or minimization['x'] is None
            or minimization['y'] is None
        ):
            raise RuntimeError(
                f"SVM optimization failed: {minimization['status']}"
            )

        # langragian mult, usually denoted as alpha in equations

        a = np.ravel(minimization['x'])

        # extract support vectors, i.e. the non-zero lagr mult
        idx = a > 1e-7
        if not np.any(idx):
            raise RuntimeError("SVM optimization produced no support vectors")

        support_a = a[idx]
        support_vectors = X[idx]
        support_vector_labels = y[idx]

        self.a = support_a
        self.support_vectors = support_vectors
        self.support_vector_labels = support_vector_labels
        # the multiplier of y.T @ alpha = 0 is the model's bias.
        
        self.intercept = float(minimization['y'][0])
        return self


    def _init_quadratic_matrix(self, y, n_samples, kernel_matrix):
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')
        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        return P, q, A, b, G, h


    def _init_kernel(self, degree, gamma, coef):
        coef = 1.0 if coef is None else coef
        # badly written code XD
        if self.kernel == "rbf":
            def g(x1, x2):
                distance = np.sum((x1 - x2) ** 2, axis=-1)
                return np.exp(-gamma * distance)
            return g
        # else: poly
        def f(x1, x2):
            return (np.sum(x1 * x2, axis=-1) + coef) ** degree
        return f

    def decision_function(self, X):
        a = self.a
        support_vectors = self.support_vectors
        support_vector_labels = self.support_vector_labels
        intercept = self.intercept
        gamma = getattr(self, "gamma_", None)
        if (
            a is None
            or support_vectors is None
            or support_vector_labels is None
            or intercept is None
            or gamma is None
        ):
            raise RuntimeError("must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != support_vectors.shape[1]:
            raise ValueError("X must have the same number of features as the fitted model")

        kernel_function = self._init_kernel(self.degree, gamma, self.coef)

        kernel_values = kernel_function(X[:, None, :], support_vectors[None, :, :])
        return kernel_values @ (a * support_vector_labels) + intercept

    def calc_cost(self, X, y):
        scores = self.decision_function(X)
        y = np.asarray(y, dtype=float)
        
        if self.a is None: 
            raise ValueError("a must not be none") 

        if self.support_vectors is None: 
            raise ValueError("support vectors cannot be empty") 

        if y.shape != scores.shape or not np.all(np.isin(y, [-1, 1])):
            raise ValueError("y must contain one label (-1 or 1) per sample")

        kernel_function = self._init_kernel(self.degree, self.gamma_, self.coef)

        vectors = self.support_vectors

        kernel_matrix = kernel_function(vectors[:, None, :], vectors[None, :, :])

        weights = self.a * self.support_vector_labels

        margin_cost = 0.5 * (weights @ kernel_matrix @ weights)

        return float(margin_cost + (self.C or 0) * np.maximum(0, 1 - y * scores).sum())

    def predict(self, X) -> np.ndarray:
        return np.sign(self.decision_function(X))
    

    # debug 
    def svm_plot(self, X, y, *, grid_size=150, show=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have exactly two features for plotting")
        self.decision_function(X)
        padding = 0.1 * np.maximum(np.ptp(X, axis=0), 1.0)
        lower = X.min(axis=0) - padding
        upper = X.max(axis=0) + padding
        xx, yy = np.meshgrid(
            np.linspace(lower[0], upper[0], grid_size),
            np.linspace(lower[1], upper[1], grid_size),
        )
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        scores = self.decision_function(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.contour(
            xx, yy, scores, levels=[-1, 0, 1],
            colors=['red', 'black', 'blue'],
            linestyles=['dotted', 'solid', 'dotted'], linewidths=1.5,
        )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f"Kernel SVM · {self.kernel.upper()}")
        fig.tight_layout()
        if show:
            plt.show()
        return ax


#
# if __name__ == "__main__":
#     X, y = make_blobs(
#         n_samples=100, centers=2, cluster_std=3.5, random_state=42
#     )
#     y = y * 2 - 1
#     # print(X.shape)
#     # print(y.shape)
#     # print(X[:10])
#     # print(y[:10])
#     svm = KernelSVM(kernel='rbf') 
#     svm.fit(X, y)
#     svm.predict(X)
#     svm.svm_plot(X, y)

