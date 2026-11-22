import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
class SVM(): 
    '''
        This class implements Soft Margin SVM 
        
        Which uses the cost function as followw: 
            min(w, b)[1/2 mag(w)^2 + C * {sum of max (0, 1 - y_i (wx_i + b)}]
            equation at rhs of plus sign is hinge loss function 
            equation at lhs of plus sign is the minimization of w
    '''
    
    def __init__(self, C=1, kernel='rbf', degree=3, n_features=2, n_iters=1000, learning_rate=0.001): 
        # we can seed here 
        self.w = np.random.rand(n_features, )
        self.b = np.random.rand() 
        self.C = C 
        self.kernel = kernel 
        self.degree = degree
        self.n_iters=n_iters
        self.learning_rate = learning_rate 
        self.history = {}
    
    
    def predict(self, X): 
        m = X.shape[0]

        y_preds = np.zeros(m) 
        # these will use the final gradient descended w and b resulting from fit. 
        for i in range(m): 
            y_preds[i] = np.dot(self.w, X[i]) + self.b
        y_preds = np.sign(y_preds) 

        return y_preds 
    
    def fit(self, X, y): 
        self.history, new_w, new_b = self._gradient_descent(X, y, self.w, self.b)

        # update to the final w and b 
        self.w, self.b = new_w, new_b  

        return self.w, self.b 

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
        
        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(self.history), list(self.history.values()), marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("SVM gradient descent history")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()

        if show:
            plt.show()
        return ax

# if __name__ == "__main__": 
#     X, y = make_blobs(
#         n_samples=100, centers=2, cluster_std=3.5, random_state=42
#     )
#     y = y * 2 - 1
#     # print(X.shape) 
#     # print(y.shape) 
#     # print(X[:10]) 
#     # print(y[:10]) 
#     svm = SVM() 
#     svm.fit(X, y)
#     svm.predict(X)
#     svm.svm_plot(X, y)
#
#

