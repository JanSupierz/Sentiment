import numpy as np
from src.models.base_model import BaseModel

class EnsembleClassifier(BaseModel):
    def __init__(self, models_dict, threshold=0.7, name="ensemble", weight=0.7):
        super().__init__(name)
        self.models = models_dict
        self.threshold = threshold
        self.weight = weight
        
    def set_weight(self, weight):
        self.weight = weight

    def train(self, X_sets, y):
        """Assumes sub-models (SVM, BERT) are already pre-trained."""
        pass

    def predict_proba(self, X_sets):
        """
        X_sets = {
            "coarse": sparse features for SVM,
            "fine": raw text for BERT
        }
        fine_weight: Weight assigned to BERT predictions for uncertain samples.
        """
        # 1. Coarse prediction (shape: [N])
        p_coarse = self.models["coarse"].predict_proba(X_sets["coarse"])
        final_probs = p_coarse.copy()

        # 2. Identify uncertain samples
        uncertain_mask = np.abs(p_coarse - 0.5) < (self.threshold - 0.5)
        uncertain_indices = np.where(uncertain_mask)[0]

        if len(uncertain_indices) > 0:
            # 3. Run BERT only on uncertain samples
            fine_inputs = [X_sets["fine"][i] for i in uncertain_indices]
            p_fine = self.models["fine"].predict_proba(fine_inputs)

            # 4. Weighted blending
            final_probs[uncertain_indices] = (self.weight * p_fine + (1 - self.weight) * p_coarse[uncertain_indices])

        return final_probs