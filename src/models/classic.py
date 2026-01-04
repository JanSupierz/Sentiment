from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.models.base_model import BaseModel
from sklearn.svm import SVC

class LinearSVMClassifier(BaseModel):
    """LinearSVC with probability calibration."""
    def __init__(self, C=1.0, name="linear_svm"):
        super().__init__(name)
        # LinearSVC by default doesn't return probabilities, so we use CalibratedClassifierCV
        self.base_model = LinearSVC(C=C, max_iter=10000, random_state=42)
        self.model = CalibratedClassifierCV(self.base_model)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]
    
class RbfSVMClassifier(BaseModel):
    """SVC with RBF kernel wrapped with probability calibration."""
    def __init__(self, C=1.0, gamma='scale', name="rbf_svm"):
        super().__init__(name)
        self.svm = SVC(C=C, kernel='rbf', gamma=gamma, probability=False, random_state=42, verbose=True)
        self.model = CalibratedClassifierCV(self.svm)
        print(f"RbfSVM '{self.name}' initialized with C={C}, gamma={gamma}")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]

class LogisticRegressionClassifier(BaseModel):
    """Standardowa Regresja Logistyczna."""
    def __init__(self, C=1.0, name="logreg"):
        super().__init__(name)
        self.model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        # Returning probability for the positive class (index 1)
        return self.model.predict_proba(X)[:, 1]