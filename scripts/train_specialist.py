import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfTransformer

from src.models.classic import LinearSVMClassifier, LogisticRegressionClassifier
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR
from src.features.builder import load_representation

logging.getLogger("transformers").setLevel(logging.ERROR)


def safe_binary_probs(probs):
    if probs.ndim == 2:
        return probs[:, 1]
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_params", default="configs/best_params.yaml")
    parser.add_argument("--bert_config", default="configs/bert.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.best_params) as f:
        cfg = yaml.safe_load(f)

    with open(PROJECT_ROOT / args.bert_config) as f:
        bert_cfg = yaml.safe_load(f)
        cfg['bert'] = bert_cfg.get('bert', {})

    MODELS_DIR.mkdir(exist_ok=True)

    # Extract features from best_params
    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']
    token_col = cfg['features'].get('token_col', 'tokens_lower')

    print(f"Loading data using token column: '{token_col}'...")

    # Load optimal sparse matrices
    X_train_sp, y_train = load_representation(token_col, nc, w, z, 'train', ngram_range=nr)
    X_val_sp, y_val = load_representation(token_col, nc, w, z, 'val', ngram_range=nr)
    X_test_sp, y_test = load_representation(token_col, nc, w, z, 'test', ngram_range=nr)

    # Load raw text for BERT to read
    df_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    X_train_txt = df_train['text_clean'].tolist()
    X_val_txt = df_val['text_clean'].tolist()
    X_test_txt = df_test['text_clean'].tolist()

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
    print(f"Base model saved to {base_model_path}.joblib")

    # MINE HARD SAMPLES (From Validation Set)
    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    print("\nMining hard samples from the validation set...")
    probs_val = safe_binary_probs(base_model.predict_proba(X_val_tfidf))
    mask_uncertain = (probs_val >= lower) & (probs_val <= upper)
    hard_indices = np.where(mask_uncertain)[0]

    print(f"Validation hard samples found: {len(hard_indices)} ({len(hard_indices)/len(y_val):.2%})")

    # Slice the validation data to create the specialist training sets
    X_hard_tfidf = X_val_tfidf[hard_indices]
    X_hard_txt = [X_val_txt[i] for i in hard_indices]
    y_hard = y_val[hard_indices]

    # TRAIN & EVALUATE SVM SPECIALIST
    print("\nTraining SVM Specialist on hard validation samples...")
    svm_spec = LinearSVMClassifier(name='SVM_Specialist')
    svm_spec.train(X_hard_tfidf, y_hard)

    # Evaluate on the full test set
    svm_spec.evaluate(X_test_tfidf, y_test, "test")
    svm_spec.save(str(MODELS_DIR / "svm_specialist"))
    print("SVM Specialist saved and evaluated.")

    # LOAD & EVALUATE BASIC BERT
    print("\nLoading and evaluating Basic BERT on test set...")
    bert_basic_path = MODELS_DIR / "bert_basic"
    if not bert_basic_path.exists():
        raise FileNotFoundError("Basic BERT model not found. Run scripts/train_bert.py first.")

    bert_basic = BERTClassifier.load(str(bert_basic_path), name="BERT_Basic")
    bert_basic.evaluate(X_test_txt, y_test, "test")
    print("Basic BERT evaluated.")

    # TRAIN & EVALUATE BERT SPECIALIST
    print("\nFine-tuning BERT Specialist on hard validation samples...")
    bert_spec = BERTClassifier.load(str(bert_basic_path), name="BERT_Specialist")
    bert_spec.freeze_backbone(num_layers_to_freeze=cfg['bert']['specialist']['num_layers_to_freeze'])

    spec_cfg = cfg['bert']['specialist']
    bert_spec.train(
        X_hard_txt, y_hard,
        epochs=int(spec_cfg['epochs']),
        batch_size=int(spec_cfg['batch_size']),
        lr=float(spec_cfg['learning_rate']),
        patience=int(spec_cfg['patience'])
    )

    # Evaluate on the full test set
    print("\nEvaluating BERT specialist on test set...")
    bert_spec.evaluate(X_test_txt, y_test, "test")
    bert_spec.save(str(MODELS_DIR / "bert_specialist"))
    print("BERT Specialist saved and evaluated.")


if __name__ == "__main__":
    main()