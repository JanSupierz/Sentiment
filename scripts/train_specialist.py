#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer

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

    MODELS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test" / "raw_predictions").mkdir(parents=True, exist_ok=True)

    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']

    print("Loading data...")
    X_train_sp, y_train = load_representation(cfg, nr, nc, w, z, 'train')
    X_val_sp, y_val = load_representation(cfg, nr, nc, w, z, 'val')
    X_test_sp, y_test = load_representation(cfg, nr, nc, w, z, 'test')

    df_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    X_train_txt = df_train['clean_review'].tolist()
    X_val_txt = df_val['clean_review'].tolist()
    X_test_txt = df_test['clean_review'].tolist()

    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_sp)
    X_val_tfidf = tfidf.transform(X_val_sp)
    X_test_tfidf = tfidf.transform(X_test_sp)

    # --- SVM ---
    print("\nTraining SVM with best parameters...")
    if cfg['model'] == 'linear_svm':
        svm = LinearSVMClassifier(C=cfg['models']['linear_svm']['C'], name='Base_SVM')
    else:
        svm = LogisticRegressionClassifier(C=cfg['models']['logreg']['C'], name='Base_LogReg')
    svm.train(X_train_tfidf, y_train)

    svm_path = MODELS_DIR / "svm_base"
    svm.save(str(svm_path))
    print(f"SVM saved to {svm_path}.joblib")

    # --- Mine hard samples from validation ---
    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    probs_val = safe_binary_probs(svm.predict_proba(X_val_tfidf))
    mask_uncertain = (probs_val >= lower) & (probs_val <= upper)
    hard_indices = np.where(mask_uncertain)[0]
    print(f"Validation hard samples: {len(hard_indices)} ({len(hard_indices)/len(y_val):.2%})")

    X_hard_txt = [X_val_txt[i] for i in hard_indices]
    y_hard = [y_val[i] for i in hard_indices]

    # --- Load Basic BERT ---
    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        raise FileNotFoundError("Basic BERT model not found. Run scripts/train_bert.py first.")
    bert_basic = BERTClassifier.load(str(bert_basic_path), name="BERT_Basic")

    # --- Train Specialist BERT ---
    print("\nFine‑tuning specialist BERT on hard samples...")
    bert_spec = BERTClassifier.load(str(bert_basic_path), name="BERT_Specialist")
    bert_spec.freeze_backbone(num_layers_to_freeze=4)

    spec_cfg = cfg['bert']['specialist']
    epochs = int(spec_cfg['epochs'])
    batch_size = int(spec_cfg['batch_size'])
    lr = float(spec_cfg['learning_rate'])
    patience = int(spec_cfg['patience'])

    bert_spec.train(
        X_hard_txt, y_hard,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience
    )

    bert_spec_path = MODELS_DIR / "bert_specialist"
    bert_spec.save(str(bert_spec_path))
    print(f"Specialist BERT saved to {bert_spec_path}")

    # --- Evaluate specialist on test set ---
    print("\nEvaluating specialist on test set...")
    probs_spec = safe_binary_probs(bert_spec.predict_proba(X_test_txt))
    preds_spec = (probs_spec > 0.5).astype(int)
    acc_spec = (preds_spec == y_test).mean()
    print(f"Specialist test accuracy: {acc_spec:.4f}")

    eval_df = pd.DataFrame({
        'true_label': y_test,
        'probability': probs_spec
    })
    eval_df.to_csv(RESULTS_DIR / "test" / "raw_predictions" / "bert_specialist.csv", index=False)
    print("Test predictions saved to results/test/raw_predictions/bert_specialist.csv")

if __name__ == "__main__":
    main()