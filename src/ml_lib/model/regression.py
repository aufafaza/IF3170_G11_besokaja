from pathlib import Path

import numpy as np

from ml_lib.model.base import BaseModel
from ml_lib.utils.functions import Sigmoid


class l1: 
    def __init__(self, alpha): 
        self.alpha = alpha 
    def __call__(self, w): 
        return self.alpha * np.abs(w).sum()

    def grad(self, w): 
        return self.alpha * np.sign(w)

class l2: 
    def __init__(self, alpha): 
        self.alpha = alpha 

    def __call__(self, w): 
        return 0.5 * self.alpha * np.dot(w, w)

    def grad(self, w): 
        return self.alpha * w 

class elastic_net: 
    def __init__(self, alpha, l1_ratio=0.5): 
        self.alpha = alpha 
        self.l1_ratio = l1_ratio 

    def __call__(self, w):
        absolute = np.abs(w).sum()
        squared = 0.5 * np.dot(w, w)
        return self.alpha * (self.l1_ratio * absolute + (1 - self.l1_ratio) * squared)

    def grad(self, w):
        l1_c = self.l1_ratio * np.sign(w) 
        l2_c = (1 - self.l1_ratio) * w 

        return self.alpha * (l1_c + l2_c) 

def save_loss_video(losses, path, *, ylabel="Training log loss"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if len(losses) == 0:
        raise ValueError("Fit the model first")

    path = Path(path)
    steps = np.arange(len(losses))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    line, = ax.plot(steps, losses, marker=".", markevery=[-1])
    ax.set(xlabel="Gradient updates completed", ylabel=ylabel,
           title="Logistic regression training")
    fig.tight_layout()

    def update(end):
        line.set_data(steps[:end], losses[:end])
        return (line,)

    frames = np.linspace(1, len(losses), min(150, len(losses)), dtype=int)
    animation = FuncAnimation(fig, update, frames=frames, repeat=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(path, writer=FFMpegWriter(fps=25, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p"]))
    finally:
        plt.close(fig)
    return path


class LogisticRegression(BaseModel): 
    def __init__(self, learning_rate=0.001, n_iters=1000, penalty=None):
        self.penalty = penalty
        self.learning_rate = learning_rate 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
        self.losses = [] 
        self.sigmoid = Sigmoid()
   
    def compute_loss(self, y, y_pred): 
        epsilon = 1e-9 
        y1 = y * np.log(y_pred + epsilon) 
        y2 = (1-y) * np.log(1 - y_pred + epsilon) 
        return -np.mean(y1 + y2)

    def feed_forward(self, X): 
        z = (X @ self.weights) + self.bias 
        A = self.sigmoid(z) 

        return A 


    def fit(self, X, y) -> BaseModel: 
        n_samples, n_features = X.shape 

        self.weights = np.zeros(n_features) 

        self.bias = 0
        self.losses = []

        for _ in range(self.n_iters): 
            A = self.feed_forward(X) 
            loss = self.compute_loss(y, A)
            dz = A - y 
            
            # gradients 
            dw = (1 / n_samples) * np.dot(X.T, dz) 
            db = (1 / n_samples) * np.sum(dz) 

            if self.penalty is not None:
                loss += self.penalty(self.weights)
                dw += self.penalty.grad(self.weights)
            self.losses.append(loss)

            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

        return self 


    def save_loss_video(self, path):
        """Export this fitted model's loss history as an MP4."""
        label = "Training log loss" if self.penalty is None else "Log loss + regularization penalty"
        return save_loss_video(self.losses, path, ylabel=label)

    def predict(self, X): 
        threshold = 0.5 
        y_hat = (X @ self.weights) + self.bias 
        y_predicted = self.sigmoid(y_hat) 
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]  
        return np.array(y_predicted_cls) 

