
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# ----------------------------
# Base class for all models
# ----------------------------
class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def train(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_label(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)
    
    def evaluate(self, X_test, y_test):
        """Simple evaluation: accuracy and classification report"""
        probs = self.predict_proba(X_test)
        preds = self.predict_label(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {self.name.upper()} EVALUATION ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=4))
        return {"accuracy": acc, "preds": preds, "probs": probs}

# ----------------------------
# LSTM Model
# ----------------------------
class LSTMClassifier(BaseModel):
    def __init__(self, vocab_size: int, max_len: int, name="lstm"):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.build_model()
        self.set_callbacks()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.max_len,)),
            Embedding(input_dim=self.vocab_size, output_dim=32),
            LSTM(32, dropout=0.2),
            Dense(16, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
        )
        self.model = model
        self.model.build(input_shape=(None, self.max_len))
        print(f"\nLSTM Model '{self.name}' Summary:")
        self.model.summary()

    def set_callbacks(self):
        self.callbacks = [
            EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
            ReduceLROnPlateau(patience=3, monitor="val_loss", factor=0.5, min_lr=1e-7)
        ]

    def train(self, X, y, epochs=20, batch=64, val_split=0.2, plot_history=True):
        print(f"\nTraining LSTM Model '{self.name}'...")
        self.history = self.model.fit(
            X, y,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch,
            callbacks=self.callbacks,
            shuffle=True,
            verbose=1
        )

        # Plot training history if requested
        if plot_history:
            self._plot_history()

    def _plot_history(self):
        """
        Plot only training and validation accuracy over epochs.
        """
        if 'accuracy' not in self.history.history:
            print("No accuracy data found in history.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).flatten()


# ----------------------------
# Linear SVM
# ----------------------------
class LinearSVMClassifier(BaseModel):
    """LinearSVC wrapped with probability calibration."""
    def __init__(self, C=1.0, name="linear_svm"):
        super().__init__(name)
        self.svm = LinearSVC(C=C, random_state=42, max_iter=10000)
        self.model = CalibratedClassifierCV(self.svm)
        print(f"LinearSVM '{self.name}' initialized with C={C}")

    def train(self, X, y):
        self.model.fit(X, y)
        print(f"LinearSVM '{self.name}' trained.")

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# ----------------------------
# Non-linear SVM (RBF kernel)
# ----------------------------
class RbfSVMClassifier(BaseModel):
    """SVC with RBF kernel wrapped with probability calibration."""
    def __init__(self, C=1.0, gamma='scale', name="rbf_svm"):
        super().__init__(name)
        self.svm = SVC(C=C, kernel='rbf', gamma=gamma, probability=False, random_state=42)
        self.model = CalibratedClassifierCV(self.svm)
        print(f"RbfSVM '{self.name}' initialized with C={C}, gamma={gamma}")

    def train(self, X, y):
        self.model.fit(X, y)
        print(f"RbfSVM '{self.name}' trained.")

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

# ----------------------------
# Logistic Regression
# ----------------------------
class LogisticRegressionClassifier(BaseModel):
    """Logistic Regression classifier, outputs probabilities."""
    def __init__(self, C=1.0, max_iter=1000, solver='lbfgs', name="logreg"):
        super().__init__(name)
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42
        )
        print(f"LogisticRegression '{self.name}' initialized with C={C}, max_iter={max_iter}, solver={solver}")

    def train(self, X, y):
        self.model.fit(X, y)
        print(f"LogisticRegression '{self.name}' trained.")

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]