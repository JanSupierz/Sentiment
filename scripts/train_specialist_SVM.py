import argparse
import yaml
import numpy as np
import scipy.sparse as sp
import logging
from sklearn.feature_extraction.text import TfidfTransformer

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.utils.paths import PROJECT_ROOT, MODELS_DIR
from src.features.builder import load_representation


def safe_binary_probs(probs):
    if probs.ndim == 2:
        return probs[:, 1]
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.best_params) as f:
        cfg = yaml.safe_load(f)

    MODELS_DIR.mkdir(exist_ok=True)

    # Extract features from best_params
    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']
    token_col = cfg['features'].get('token_col', 'tokens_lower')

    print(f"Loading sparse data using token column: '{token_col}'...")

    # Load optimal sparse matrices
    X_train_sp, y_train = load_representation(token_col, nc, w, z, 'train', ngram_range=nr)
    X_val_sp, y_val = load_representation(token_col, nc, w, z, 'val', ngram_range=nr)
    X_test_sp, y_test = load_representation(token_col, nc, w, z, 'test', ngram_range=nr)

    print("Applying TF-IDF...")
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_sp)
    X_val_tfidf = tfidf.transform(X_val_sp)
    X_test_tfidf = tfidf.transform(X_test_sp)

    # TRAIN & EVALUATE BASE MODEL
    base_model_type = cfg['model']
    print(f"\nTraining base model ({base_model_type}) with best parameters...")

    if base_model_type == 'linear_svm':
        base_model = LinearSVMClassifier(name='Base_SVM')
        model_filename = "svm_base"
    elif base_model_type == 'logreg':
        base_model = LogisticRegressionClassifier(name='Base_LogReg')
        model_filename = "logreg_base"
    else:
        raise ValueError(f"Unknown model type in config: {base_model_type}")

    base_model.train(X_train_tfidf, y_train)
    base_model.evaluate(X_test_tfidf, y_test, "test")

    base_model_path = MODELS_DIR / model_filename
    base_model.save(str(base_model_path))
    print(f"Base model saved to {base_model_path}")

    # MINE HARD SAMPLES (From Train + Validation Sets)
    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    print("\nMining hard samples from the TRAIN and VALIDATION sets...")
    
    # 1. Mine Train Hard Samples
    probs_train = safe_binary_probs(base_model.predict_proba(X_train_tfidf))
    mask_uncertain_train = (probs_train >= lower) & (probs_train <= upper)
    X_hard_train = X_train_tfidf[mask_uncertain_train]
    y_hard_train = y_train[mask_uncertain_train]
    
    print(f"Train hard samples found: {mask_uncertain_train.sum()} ({mask_uncertain_train.sum()/len(y_train):.2%})")

    # 2. Mine Validation Hard Samples
    probs_val = safe_binary_probs(base_model.predict_proba(X_val_tfidf))
    mask_uncertain_val = (probs_val >= lower) & (probs_val <= upper)
    X_hard_val = X_val_tfidf[mask_uncertain_val]
    y_hard_val = y_val[mask_uncertain_val]
    
    print(f"Validation hard samples found: {mask_uncertain_val.sum()} ({mask_uncertain_val.sum()/len(y_val):.2%})")

    # TRAIN & EVALUATE SVM SPECIALIST
    print(f"\nTraining SVM Specialist on {len(y_hard_val)} validation hard samples...")
    svm_spec = LinearSVMClassifier(name='SVM_Specialist')
    svm_spec.train(X_hard_val, y_hard_val)

    # Evaluate on the full test set
    svm_spec.evaluate(X_test_tfidf, y_test, "test")
    svm_spec.save(str(MODELS_DIR / "svm_specialist"))
    
    # Save indices for both sets
    np.save(MODELS_DIR / "hard_indices_train.npy", np.where(mask_uncertain_train)[0])
    np.save(MODELS_DIR / "hard_indices_val.npy", np.where(mask_uncertain_val)[0])
    print("SVM Specialist trained and saved. Hard sample indices saved for both sets.")


if __name__ == "__main__":
    main()