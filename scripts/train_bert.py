#!/usr/bin/env python3
import argparse
import yaml
import pandas as pd
import numpy as np
from src.models.deep import BERTClassifier
from src.utils.paths import PROJECT_ROOT, DATA_DIR, MODELS_DIR
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print("\n>>> Training Basic BERT Baseline <<<")

    df_train = pd.read_parquet(DATA_DIR / "preprocessed" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "preprocessed" / "val.parquet")

    X_train_txt = df_train['clean_review'].tolist()
    y_train = df_train['sentiment'].tolist()
    X_val_txt = df_val['clean_review'].tolist()
    y_val = df_val['sentiment'].values.astype(np.int32)

    bert_cfg = cfg['bert']['basic']
    
    # Init with a descriptive name. BaseModel uses this to save the raw_predictions CSV
    model = BERTClassifier(
        model_name=bert_cfg['model_name'], 
        max_len=bert_cfg['max_len'], 
        name='bert_basic_baseline' 
    )
    
    model.train(
        X_train_txt, y_train, 
        epochs=bert_cfg['epochs'], 
        batch_size=bert_cfg['batch_size'],
        lr=float(bert_cfg['learning_rate']),
        patience=bert_cfg['patience']
    )

    # Automatically saves to results/val/raw_predictions/bert_basic_baseline.csv
    model.evaluate(X_val_txt, y_val, name="val")
    
    # FIX: Explicitly define the destination variable so the print statement works
    bert_dest = MODELS_DIR / "bert_basic"
    model.save(str(bert_dest))
    print(f"BERT Model saved to {bert_dest}")

if __name__ == "__main__":
    main()