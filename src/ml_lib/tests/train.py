from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_lib.model.regression import LogisticRegression, elastic_net

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    model = LogisticRegression(learning_rate=0.1, penalty=elastic_net(alpha=0.01))
    model.fit(X_train, y_train)

    path = Path(__file__).resolve().parents[3] / "output" / "logistic_loss_elastic_net.mp4"
    model.save_loss_video(path)
    print(f"Saved training loss video to {path}")
