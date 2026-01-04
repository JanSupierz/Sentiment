import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from IPython.display import display, Markdown
from src.utils.visualizer import ModelVisualizer
import joblib
import os

class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        """Standardized save for Scikit-Learn based models."""
        if not path.endswith('.joblib'): path += '.joblib'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Classic model saved to: {path}")

    @classmethod
    def load(cls, path: str):
        """Standardized load for Scikit-Learn based models."""
        if not path.endswith('.joblib'): path += '.joblib'
        return joblib.load(path)
    
    def train(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_label(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)
    
    def evaluate(self, X_test, y_test, lower_threshold: float, name: str = "Train Data"):
        probs = self.predict_proba(X_test)
        preds = self.predict_label(X_test)
        acc = accuracy_score(y_test, preds)

        display(Markdown(f"## {self.name.upper()} Evaluation ({name})"))
        display(Markdown(f"**Accuracy:** {acc:.4f}"))

        report_dict = classification_report(y_test, preds, output_dict=True, digits=4)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(5)
        report_df['support'] = report_df['support'].astype(int)

        display(report_df)
        ModelVisualizer.plot_confusion_matrix(y_test, preds, f'Confusion Matrix {self.name} ({name})')
        ModelVisualizer.plot_certainty_histogram(preds, probs, y_test, lower_threshold, f'Certainty Distribution {self.name} ({name})')
        stats_table = ModelVisualizer.get_detailed_certainty_stats(preds, probs, y_test, lower=lower_threshold)
        display(stats_table)
        return probs, preds