from collections import Counter
from tqdm.auto import tqdm
from src.utils.loader import DataLoader 

def process_evaluation_set(
    dataset, 
    set_name, 
    train_mapping, 
    stop_units_set, 
    min_freq, 
    important_set, 
    n_gram_range=(1, 3),
    extractor_obj=None,
    cluster_centers=None,
    printing=True
):
    """
    Filters and maps IDs.
    """
    if printing: print(f"\n1/3: Scanning {set_name} for local rare tokens...")

    set_counts = Counter()
    for item in tqdm(dataset, desc=f"Scanning {set_name}", disable=not printing):
        units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)
        valid_units = [u for u in units if u not in stop_units_set]
        set_counts.update(valid_units)
    
    # These are tokens that appear enough times in the current set to be considered
    significant_units = {u for u, count in set_counts.items() if count >= min_freq}
    
    # Calculate how many of these are actually "new" (Unknown)
    all_unknown_units = sorted(list({u for u in significant_units if u not in train_mapping}))
    num_significant = len(significant_units)
    num_unknown = len(all_unknown_units)
    unknown_pct = (num_unknown / num_significant * 100) if num_significant > 0 else 0

    if printing: print(f"2/3: Mapping {num_unknown} unknown units ({unknown_pct:.1f}% of significant) for {set_name}...")
    
    unknown_mapping = {}
    if extractor_obj and cluster_centers is not None and all_unknown_units: 
        unknown_mapping = extractor_obj.map_units_to_clusters(all_unknown_units, cluster_centers, printing = False)

    if printing: print(f"3/3: Mapping and Filtering {set_name}...")
        
    stats = {"from_train": 0, "from_unknown": 0, "total_filtered_out": 0}

    for item in tqdm(dataset, desc=f"Processing {set_name}", disable=not printing):
        units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)
        filtered_ids = []
        
        for u in units:
            if u in stop_units_set or u not in significant_units:
                stats["total_filtered_out"] += 1
                continue
            
            cid = None
            is_new = False
            
            if u in train_mapping:
                cid = train_mapping[u]
            elif u in unknown_mapping:
                cid = unknown_mapping[u]
                is_new = True
            
            if cid is not None and cid in important_set:
                filtered_ids.append(cid)
                if is_new: stats["from_unknown"] += 1
                else: stats["from_train"] += 1
            else:
                stats["total_filtered_out"] += 1
        
        item['important_ids'] = filtered_ids

    if printing:
        total_mapped = stats["from_train"] + stats["from_unknown"]
        print(f"\n--- Mapping Verification for {set_name} ---")
        print(f"Significant Vocabulary Discovery:")
        print(f"  - Total Significant Units: {num_significant}")
        print(f"  - Known (from train):     {num_significant - num_unknown}")
        print(f"  - Unknown (New):          {num_unknown} ({unknown_pct:.1f}%)")
        
        if total_mapped > 0:
            print(f"\nToken-Level Impact (Instances in Text):")
            print(f"  - Units from Train Mapping:   {stats['from_train']} ({(stats['from_train']/total_mapped)*100:.1f}%)")
            print(f"  - Units from Unknown Mapping: {stats['from_unknown']} ({(stats['from_unknown']/total_mapped)*100:.1f}%)")
            print(f"  - Successfully re-mapped:     {len(unknown_mapping)}/{num_unknown} unique unknown tokens.")
        else:
            print(f"\n--- Warning: No units from {set_name} were mapped to important concepts ---")