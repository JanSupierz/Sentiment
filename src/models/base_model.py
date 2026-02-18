import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report
from src.utils.paths import RESULTS_DIR

class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        """Standardized save for Scikit-Learn based models."""
        if not path.endswith('.joblib'):
            path += '.joblib'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Standardized load for Scikit-Learn based models."""
        if not path.endswith('.joblib'):
            path += '.joblib'
        return joblib.load(path)

    def train(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_label(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def get_evaluation_data(self, y_true, probs) -> pd.DataFrame:
        return pd.DataFrame({
            'true_label': y_true,
            'probability': probs
        })

    def evaluate(self, X_test, y_test, name: str = "val"):
        probs = self.predict_proba(X_test)
        preds = self.predict_label(X_test)
        
        # --- PATHS ---
        data_dir = RESULTS_DIR / name / "raw_predictions"
        report_dir = RESULTS_DIR / name / "classification_reports"

        for d in [data_dir, report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Save Raw Evaluation Data (Source for all future plots)
        eval_df = self.get_evaluation_data(y_test, probs)
        eval_df.to_csv(data_dir / f"{self.name}.csv", index=False)

        # 2. Save Standard Metrics
        report_dict = classification_report(y_test, preds, output_dict=True, digits=4)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(report_dir / f"{self.name}.csv")