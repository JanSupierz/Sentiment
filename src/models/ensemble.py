import numpy as np
from src.models.base_model import BaseModel


def safe_binary_probs(probs):
    """
    Ensures probabilities are a 1D array of positive-class probabilities.
    Handles Sklearn (N, 2) and TF/Keras (N, 1) or (N,) shapes.
    """
    if probs.ndim == 2:
        # If shape is (N, 2), take the second column (positive class)
        # If shape is (N, 1), flatten it
        return probs[:, 1] if probs.shape[1] > 1 else probs.ravel()
    return probs


class EnsembleClassifier(BaseModel):
    
    def __init__(self, models_dict, delegation_threshold=0.3, name="Cascade_Ensemble", specialist_weight=1.0):

        super().__init__(name)
        self.models = models_dict
        self.lower = delegation_threshold
        self.upper = 1.0 - delegation_threshold
        self.specialist_weight = specialist_weight

    def train(self, X_sets, y):
        raise NotImplementedError("Train sub-models using train_bert.py or train_specialist.py first.")

    def predict_proba(self, X_sets):
        coarse_model = self.models["coarse"]
        if hasattr(coarse_model, "predict_proba"):
            p_coarse_raw = coarse_model.predict_proba(X_sets["coarse"])
        else:
            p_coarse_raw = coarse_model.model.predict_proba(X_sets["coarse"])

        p_coarse = safe_binary_probs(p_coarse_raw)
        final_probs = p_coarse.copy()

        uncertain_mask = (p_coarse >= self.lower) & (p_coarse <= self.upper)
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) > 0:
            fine_inputs = [X_sets["fine"][i] for i in uncertain_indices]
            p_fine_raw = self.models["fine"].predict_proba(fine_inputs)
            p_fine = safe_binary_probs(p_fine_raw)

            w = self.specialist_weight
            final_probs[uncertain_indices] = (w * p_fine) + ((1 - w) * p_coarse[uncertain_indices])

        return final_probs

    def predict_label(self, X_sets):
        probs = self.predict_proba(X_sets)
        return (probs > 0.5).astype(int)