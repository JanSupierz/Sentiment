import numpy as np
from models import LSTMClassifier, LinearSVMClassifier, RbfSVMClassifier, LogisticRegressionClassifier
from visualization import ModelVisualizer

class EnsemblePredictor:
    @staticmethod
    def predict(models, X_sets, threshold=0.7):
        p_all = models["all"].predict_proba(X_sets["all"])
        p_pos = models["pos"].predict_proba(X_sets["pos"])
        p_neg = models["neg"].predict_proba(X_sets["neg"])

        final_preds = np.zeros_like(p_all, dtype=int)
        final_probs = np.zeros_like(p_all)

        for i in range(len(p_all)):
            neg_prob = 1 - p_neg[i]
            if p_pos[i] > threshold and neg_prob > threshold:
                final_preds[i] = int(p_all[i] > 0.5)
                final_probs[i] = p_all[i]
            elif p_pos[i] > threshold:
                final_preds[i] = 1
                final_probs[i] = p_pos[i] #High probability, meaning positive classification
            elif neg_prob > threshold:
                final_preds[i] = 0
                final_probs[i] = p_neg[i] #Low probablity, meaning negative classification
            else:
                final_preds[i] = int(p_all[i] > 0.5)
                final_probs[i] = p_all[i]

        return final_preds, final_probs

def run_experiment(
    X_train_sets, y_train, 
    X_test_sets, y_test,
    texts,
    model_type,
    vocab_size=None,
    seq_lens=None,
    lower_uncertain=0.3,
    upper_uncertain=0.7,
    n_examples=10
):
    """
    X_train_sets, X_test_sets: dict of 'all', 'pos', 'neg', 'important' datasets
    y_train, y_test: labels
    texts: optional, pandas Series of review texts corresponding to X_test_sets["all"]
    """
    models = {}
    for key in ["all", "pos", "neg", "important"]:
        if model_type == "lstm":
            models[key] = LSTMClassifier(vocab_size=vocab_size, max_len=seq_lens[key], name=key)
        elif model_type == "linear-svm":
            models[key] = LinearSVMClassifier(name=key)
        elif model_type == "rbf-svm":
            models[key] = RbfSVMClassifier(name=key)
        elif model_type == "lr":
            models[key] = LogisticRegressionClassifier(name=key)
        else:
            raise ValueError("Unknown model type")

        # Train & evaluate
        models[key].train(X_train_sets[key], y_train)
        models[key].evaluate(X_test_sets[key], y_test)

        # Predict & plot
        probs = models[key].predict_proba(X_test_sets[key])
        preds = models[key].predict_label(X_test_sets[key])
        print(f"\n=== {key.upper()} MODEL ===")
        print(f"Accuracy: {np.mean(preds == y_test):.4f}")
        ModelVisualizer.plot_confusion_matrix(y_test, preds, f"{key.upper()} Confusion Matrix")
        ModelVisualizer.plot_certainty_analysis(probs, y_test, f"{key.upper()} Certainty vs Correctness")

        print(f"\n{key.upper()} MODEL - Unsure Predictions (probability near {lower_uncertain}-{upper_uncertain}):")
        # Unsure predictions
        unsure_idx = np.where((probs >= lower_uncertain) & (probs <= upper_uncertain))[0]
        for i in unsure_idx[:n_examples]:
            print(f"Probability [{probs[i]:.2f}], Label [{y_test.iloc[i]}]: \"{texts.iloc[i]}\"")

        print(f"\n{key.upper()} MODEL - Misclassified Predictions:")
        # Misclassified predictions
        mis_idx = np.where(preds != y_test)[0]
        for i in mis_idx[:n_examples]:
            print(f"Probability [{probs[i]:.2f}], Label [{y_test.iloc[i]}], Pred [{preds[i]}]: \"{texts.iloc[i]}\"")

    # Ensemble
    ens_preds, ens_probs = EnsemblePredictor.predict(models, X_test_sets)
    acc = np.mean(ens_preds == y_test)
    print(f"\nENSEMBLE ACCURACY: {acc:.4f}")
    ModelVisualizer.plot_confusion_matrix(y_test, ens_preds, "Ensemble Confusion Matrix")
    ModelVisualizer.plot_certainty_analysis(ens_probs, y_test, "Ensemble Certainty vs Correctness")

    # Ensemble unsure / misclassified examples
    if texts is not None:
        print(f"\nENSEMBLE - Unsure Predictions (probability near {lower_uncertain}-{upper_uncertain}):")
        unsure_idx = np.where((ens_probs >= lower_uncertain) & (ens_probs <= upper_uncertain))[0]
        for i in unsure_idx[:n_examples]:
            print(f"Probability [{ens_probs[i]:.2f}], Label [{y_test.iloc[i]}]: \"{texts.iloc[i]}\"")

        print(f"\nENSEMBLE - Misclassified Predictions:")
        mis_idx = np.where(ens_preds != y_test)[0]
        for i in mis_idx[:n_examples]:
            print(f"Probability [{ens_probs[i]:.2f}], Label [{y_test.iloc[i]}], Pred [{ens_preds[i]}]: \"{texts.iloc[i]}\"")

    return {
        "models": models,
        "ensemble_preds": ens_preds,
        "ensemble_probs": ens_probs,
        "ensemble_acc": acc
    }
