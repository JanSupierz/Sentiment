import pandas as pd
from src.utils.paths import RESULTS_DIR

def save_to_central_csv(results):
    """
    Saves a dictionary or list of dictionaries containing experiment metrics 
    to a central CSV file, replacing older runs of the same configuration.
    """
    csv_path = RESULTS_DIR / "experiment_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
        
    df_new = pd.DataFrame(results)
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
        
        subset_cols = ['model', 'ngram_range', 'n_concepts', 'z_threshold', 'sentiment_weight']
        
        for col in subset_cols:
            if col not in df.columns:
                df[col] = "N/A"
        
        df[subset_cols] = df[subset_cols].astype(str)
        df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
    else:
        df = df_new
        
    df.to_csv(csv_path, index=False)
    print(f"Metrics successfully saved to {csv_path}")