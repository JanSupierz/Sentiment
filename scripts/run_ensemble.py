#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.features.builder import load_representation

logging.getLogger("transformers").setLevel(logging.ERROR)

def safe_binary_probs(probs):
    if probs.ndim == 2:
        return probs[:, 1]
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / args.best_params) as f:
        best = yaml.safe_load(f)

    cfg.setdefault('features', {})
    cfg.setdefault('models', {})
    cfg.setdefault('model', 'linear_svm')
    cfg.setdefault('cascade', {})

    cfg['features'].update(best.get('features', {}))
    cfg['model'] = best.get('model', cfg['model'])
    if 'models' in best:
        cfg['models'].update(best['models'])
    if 'cascade' in best:
        cfg['cascade'].update(best['cascade'])

    threshold = cfg['cascade']['delegation_threshold']

    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']

    # Load test data – will automatically build test matrix if missing
    X_test_sp, y_test = load_representation(cfg, nr, nc, w, z, 'test')
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")
    X_test_txt = df_test['clean_review'].tolist()

    # TF‑IDF
    X_train_sp, _ = load_representation(cfg, nr, nc, w, z, 'train')
    tfidf = TfidfTransformer()
    tfidf.fit(X_train_sp)
    X_test_tfidf = tfidf.transform(X_test_sp)

    # Load models
    svm_path = MODELS_DIR / "svm_base.joblib"
    if not svm_path.exists():
        raise FileNotFoundError("SVM model not found. Run scripts/train_specialized.py first.")
    svm = joblib.load(svm_path)

    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        raise FileNotFoundError("Basic BERT not found. Run scripts/train_bert.py first.")
    bert_basic = BERTClassifier.load(str(bert_basic_path), name="BERT_Basic")

    bert_spec_path = MODELS_DIR / "bert_specialist"
    if not bert_spec_path.exists():
        raise FileNotFoundError("Specialist BERT not found. Run scripts/train_specialized.py first.")
    bert_spec = BERTClassifier.load(str(bert_spec_path), name="BERT_Specialist")

    # Predictions
    probs_svm = safe_binary_probs(svm.predict_proba(X_test_tfidf))
    preds_svm = (probs_svm > 0.5).astype(int)

    probs_basic = safe_binary_probs(bert_basic.predict_proba(X_test_txt))
    preds_basic = (probs_basic > 0.5).astype(int)

    probs_spec = safe_binary_probs(bert_spec.predict_proba(X_test_txt))
    preds_spec = (probs_spec > 0.5).astype(int)

    # Cascade
    lower = threshold
    upper = 1.0 - lower
    uncertain_mask = (probs_svm >= lower) & (probs_svm <= upper)
    delegation_rate = uncertain_mask.mean()

    preds_cascade_basic = preds_svm.copy()
    preds_cascade_basic[uncertain_mask] = preds_basic[uncertain_mask]

    preds_cascade_spec = preds_svm.copy()
    preds_cascade_spec[uncertain_mask] = preds_spec[uncertain_mask]

    # Accuracies
    acc_svm = accuracy_score(y_test, preds_svm)
    acc_basic = accuracy_score(y_test, preds_basic)
    acc_spec = accuracy_score(y_test, preds_spec)
    acc_cascade_basic = accuracy_score(y_test, preds_cascade_basic)
    acc_cascade_spec = accuracy_score(y_test, preds_cascade_spec)

    # Save results
    results = {
        "SVM": acc_svm,
        "Basic BERT": acc_basic,
        "Specialist BERT": acc_spec,
        "SVM+Basic Cascade": acc_cascade_basic,
        "SVM+Specialist Cascade": acc_cascade_spec,
        "Delegation Rate": delegation_rate,
        "Threshold": threshold
    }
    results_df = pd.DataFrame([results])
    out_dir = RESULTS_DIR / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "ensemble_results.csv", index=False)

    print("\n=== CASCADE ENSEMBLE RESULTS ===")
    for name, acc in results.items():
        if "Rate" not in name and "Threshold" not in name:
            print(f"{name:25s}: {acc:.4f}")
    print(f"Delegation rate        : {delegation_rate:.2%}")
    print(f"Threshold used         : {threshold:.3f}")
    print(f"\nResults saved to {out_dir / 'ensemble_results.csv'}")

if __name__ == "__main__":
    main()