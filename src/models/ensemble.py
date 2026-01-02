import numpy as np
from src.models.base_model import BaseModel

class EnsembleClassifier(BaseModel):
    def __init__(self, models_dict, threshold=0.7, name="ensemble"):
        super().__init__(name)
        self.models = models_dict
        self.threshold = threshold 

    def train(self, X_sets, y):
        """Assumes sub-models (SVM, BERT) are already pre-trained."""
        pass 

    def predict_proba(self, X_sets):
        """
        X_sets must be a dictionary:
        {
            "coarse": Sparse matrix for SVM,
            "fine": Raw text for BERT,
        }
        """
        # 1. Run the coarse model on all inputs
        p_coarse = self.models["coarse"].predict_proba(X_sets["coarse"])
        
        # 2. Initialize final probabilities with coarse results
        final_probs = p_coarse.copy()
        
        # 3. Identify indices where the coarse model "didn't work" (not sure enough)
        # Confidence is the distance from the 0.5 decision boundary
        uncertain_mask = np.abs(p_coarse - 0.5) < (self.threshold - 0.5)
        uncertain_indices = np.where(uncertain_mask)[0]
        
        if len(uncertain_indices) > 0:
            # 4. ONLY predict fine for the uncertain subset
            # Filter the raw text inputs for BERT
            fine_inputs = [X_sets["fine"][i] for i in uncertain_indices]
            p_fine_subset = self.models["fine"].predict_proba(fine_inputs)
            
            # 5. Inject fine predictions back into the final results
            final_probs[uncertain_indices] = p_fine_subset
                
        return final_probs