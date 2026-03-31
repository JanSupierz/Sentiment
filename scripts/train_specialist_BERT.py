import argparse
import yaml
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

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

    # Extract features needed to recreate TF-IDF for hard sample mining
    nr = cfg['features']['ngram_range']
    nc = cfg['features']['n_concepts']
    w = cfg['features']['sentiment_weight']
    z = cfg['features']['z_threshold']
    token_col = cfg['features'].get('token_col', 'tokens_lower')

    # Load optimal sparse matrices and fit TF-IDF to get validation probabilities
    print("Loading data for hard sample mining...")
    X_train_sp, _ = load_representation(token_col, nc, w, z, 'train', ngram_range=nr)
    X_val_sp, y_val = load_representation(token_col, nc, w, z, 'val', ngram_range=nr)
    _, y_test = load_representation(token_col, nc, w, z, 'test', ngram_range=nr)

    tfidf = TfidfTransformer()
    tfidf.fit(X_train_sp) # Fit on train to transform val
    X_val_tfidf = tfidf.transform(X_val_sp)

    # Load raw text for BERT to read
    print("Loading raw text data...")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "preprocessed" / "test.parquet")

    X_val_txt = df_val['text_clean'].tolist()
    X_test_txt = df_test['text_clean'].tolist()

    # LOAD BASE MODEL TO MINE HARD SAMPLES
    base_model_type = cfg['model']
    model_filename = "svm_base" if base_model_type == 'linear_svm' else "logreg_base"
    base_model_path = MODELS_DIR / model_filename

    print(f"\nLoading base model from {base_model_path}...")
    try:
        base_model = joblib.load(f"{base_model_path}.joblib")
    except FileNotFoundError:
        raise FileNotFoundError(f"Base model not found. Run train_svm.py first.")

    threshold = cfg['cascade']['delegation_threshold']
    lower = threshold
    upper = 1.0 - lower

    print("Mining hard samples from the validation set...")
    probs_val = safe_binary_probs(base_model.predict_proba(X_val_tfidf))
    mask_uncertain = (probs_val >= lower) & (probs_val <= upper)
    hard_indices = np.where(mask_uncertain)[0]

    print(f"Validation hard samples found: {len(hard_indices)} ({len(hard_indices)/len(y_val):.2%})")

    # Slice the validation data to create the specialist training sets
    X_hard_txt = [X_val_txt[i] for i in hard_indices]
    y_hard = y_val[hard_indices]

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