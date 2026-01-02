from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.models.base_model import BaseModel

class LinearSVMClassifier(BaseModel):
    """LinearSVC z kalibracją prawdopodobieństwa."""
    def __init__(self, C=1.0, name="linear_svm"):
        super().__init__(name)
        # LinearSVC domyślnie nie zwraca prawdopodobieństw, dlatego używamy CalibratedClassifierCV
        self.base_model = LinearSVC(C=C, max_iter=10000, random_state=42)
        self.model = CalibratedClassifierCV(self.base_model)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Zwracamy prawdopodobieństwo dla klasy pozytywnej (indeks 1)
        return self.model.predict_proba(X)[:, 1]

class LogisticRegressionClassifier(BaseModel):
    """Standardowa Regresja Logistyczna."""
    def __init__(self, C=1.0, name="logreg"):
        super().__init__(name)
        self.model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Zwracamy prawdopodobieństwo dla klasy pozytywnej (indeks 1)
        return self.model.predict_proba(X)[:, 1]