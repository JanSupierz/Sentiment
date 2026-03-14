import argparse
import yaml
from src.utils.paths import PROJECT_ROOT
from src.features import builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Get parameter spaces from config
    grid = cfg['grid_search']
    n_concepts_list = grid['n_concepts']
    sentiment_weights = grid['sentiment_weight']

    print("=== PHASE 1: Build unit matrices ===")
    builder.build_unit_matrices('tokens_lower', max_df=0.7)

    print("\n=== PHASE 2: Compute unit Z‑score indices ===")
    builder.compute_unit_z_indices('tokens_lower', [2.0])

    print("\n=== PHASE 3: Compute embeddings (needed for concepts) ===")
    builder.compute_embeddings('tokens_lower')

    print("\n=== PHASE 4: Extract concepts for each (k, w) combination ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.extract_concepts('tokens_lower', nc, w)

    print("\n=== PHASE 5: Build concept matrices ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.build_concept_matrices('tokens_lower', nc, w)

    print("\n=== PHASE 6: Compute concept Z‑score indices ===")
    for nc in n_concepts_list:
        if nc == 0:
            continue
        for w in sentiment_weights:
            builder.compute_concept_z_indices('tokens_lower', nc, w, [2.0])


if __name__ == "__main__":
    main()