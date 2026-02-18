# src/models/ensemble.py
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
        """
        Parameters:
        -----------
        models_dict : dict
            Contains "coarse" (SVM/LogReg) and "fine" (BERT) model instances.
        delegation_threshold : float
            The certainty threshold (e.g., 0.3 means delegating if 0.3 < p < 0.7).
        specialist_weight : float
            Blending weight. Default 1.0 means BERT completely replaces SVM predictions
            on delegated samples.
        """
        super().__init__(name)
        self.models = models_dict
        self.lower = delegation_threshold
        self.upper = 1.0 - delegation_threshold
        self.specialist_weight = specialist_weight

    def train(self, X_sets, y):
        """
        In this pipeline, sub-models are trained independently via specialized scripts.
        """
        raise NotImplementedError("Train sub-models using train_bert.py or train_specialist.py first.")

    def predict_proba(self, X_sets):
        """
        Routes samples based on coarse model certainty.
        
        Parameters:
        -----------
        X_sets : dict
            {
                "coarse": sparse features/TF-IDF for SVM,
                "fine": raw text list for BERT
            }
        """
        # 1. Get initial predictions from Coarse Model (SVM)
        # Check if it's a wrapped Scikit-Learn model or direct
        coarse_model = self.models["coarse"]
        if hasattr(coarse_model, "predict_proba"):
            p_coarse_raw = coarse_model.predict_proba(X_sets["coarse"])
        else:
            p_coarse_raw = coarse_model.model.predict_proba(X_sets["coarse"])
            
        p_coarse = safe_binary_probs(p_coarse_raw)
        final_probs = p_coarse.copy()

        # 2. Identify uncertain samples (Delegation Zone)
        # Unified logic: Delegate if p is between lower and upper threshold
        uncertain_mask = (p_coarse >= self.lower) & (p_coarse <= self.upper)
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) > 0:
            # 3. Run Fine Model (BERT) ONLY on the uncertain samples
            fine_inputs = [X_sets["fine"][i] for i in uncertain_indices]
            p_fine_raw = self.models["fine"].predict_proba(fine_inputs)
            p_fine = safe_binary_probs(p_fine_raw)

            # 4. Sequential Replacement / Blending
            # If specialist_weight = 1.0, SVM is ignored for these samples
            w = self.specialist_weight
            final_probs[uncertain_indices] = (w * p_fine) + ((1 - w) * p_coarse[uncertain_indices])

        return final_probs

    def predict_label(self, X_sets):
        """Returns 0 or 1 based on final blended probabilities."""
        probs = self.predict_proba(X_sets)
        return (probs > 0.5).astype(int)