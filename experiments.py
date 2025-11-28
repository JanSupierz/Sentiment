import numpy as np
import pandas as pd
from IPython.display import display, Markdown
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
                final_probs[i] = p_pos[i]

            elif neg_prob > threshold:
                final_preds[i] = 0
                final_probs[i] = p_neg[i]

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
    n_examples=5
):
    upper_uncertain = 1 - lower_uncertain

    models = {}
    results = {}

    for key in ["all", "pos", "neg", "important"]:

        # ------------------------------------------
        # Select model
        # ------------------------------------------
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

        # ------------------------------------------
        # Train & evaluate
        # ------------------------------------------
        models[key].train(X_train_sets[key], y_train)
        probs, preds = models[key].evaluate(X_test_sets[key], y_test, lower_uncertain)

        # ------------------------------------------
        # Unsure predictions DataFrame
        # ------------------------------------------
        unsure_idx = np.where((probs >= lower_uncertain) & (probs <= upper_uncertain))[0]
        unsure_df = pd.DataFrame({
            "probability": probs[unsure_idx],
            "true_label": y_test.iloc[unsure_idx].values,
            "text": texts.iloc[unsure_idx].values
        })

        # ------------------------------------------
        # Misclassified DataFrame
        # ------------------------------------------
        mis_idx = np.where(preds != y_test)[0]
        mis_df = pd.DataFrame({
            "probability": probs[mis_idx],
            "true_label": y_test.iloc[mis_idx].values,
            "predicted": preds[mis_idx],
            "text": texts.iloc[mis_idx].values
        })

        # ------------------------------------------
        # Show results in notebook
        # ------------------------------------------
        display(Markdown(f"## {key.upper()} Model — Unsure Predictions"))
        display(unsure_df.head(n_examples))

        display(Markdown(f"## {key.upper()} Model — Misclassified Predictions"))
        display(mis_df.head(n_examples))

        # ------------------------------------------
        # Store results
        # ------------------------------------------
        results[key] = {
            "preds": preds,
            "probs": probs,
            "unsure_df": unsure_df,
            "misclassified_df": mis_df
        }

    # ---------------------------------------------------------------------
    # Ensemble Section
    # ---------------------------------------------------------------------
    ens_preds, ens_probs = EnsemblePredictor.predict(models, X_test_sets, threshold=upper_uncertain)
    ensemble_acc = np.mean(ens_preds == y_test)

    display(Markdown(f"# Ensemble Accuracy: **{ensemble_acc:.4f}**"))

    ModelVisualizer.plot_confusion_matrix(y_test, ens_preds, "Ensemble")
    ModelVisualizer.plot_certainty_analysis(ens_preds, ens_probs, y_test, lower_uncertain, "Ensemble")

    # Unsure
    unsure_idx = np.where((ens_probs >= lower_uncertain) & (ens_probs <= upper_uncertain))[0]
    ensemble_unsure_df = pd.DataFrame({
        "probability": ens_probs[unsure_idx],
        "true_label": y_test.iloc[unsure_idx].values,
        "text": texts.iloc[unsure_idx].values
    })

    # Misclassified
    mis_idx = np.where(ens_preds != y_test)[0]
    ensemble_mis_df = pd.DataFrame({
        "probability": ens_probs[mis_idx],
        "true_label": y_test.iloc[mis_idx].values,
        "predicted": ens_preds[mis_idx],
        "text": texts.iloc[mis_idx].values
    })

    display(Markdown("## Ensemble — Unsure Predictions"))
    display(ensemble_unsure_df.head(n_examples))

    display(Markdown("## Ensemble — Misclassified Predictions"))
    display(ensemble_mis_df.head(n_examples))

    # ------------------------------------------
    # Verification: no identical rows in TF-IDF train/test
    # ------------------------------------------
    display(Markdown("## Train/Test Feature Overlap Check"))

    overlap_report = {}

    for key in ["all", "pos", "neg", "important"]:
        X_train = X_train_sets[key]  # csr_matrix
        X_test = X_test_sets[key]    # csr_matrix

        # Convert each row to bytes for hashing
        train_hashes = {hash(row.todense().tobytes()) for row in X_train}
        test_hashes = {hash(row.todense().tobytes()) for row in X_test}

        # Compute intersection
        overlap = train_hashes & test_hashes
        overlap_count = len(overlap)
        overlap_report[key] = overlap_count

        if overlap_count > 0:
            display(Markdown(f"**WARNING:** {overlap_count} identical TF-IDF rows detected in '{key}' train/test sets."))
        else:
            display(Markdown(f"✅ No identical TF-IDF rows detected in '{key}'"))

    # Summary table
    overlap_df = pd.DataFrame.from_dict(overlap_report, orient='index', columns=['train_test_overlap'])
    display(overlap_df)