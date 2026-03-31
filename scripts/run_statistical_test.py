import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from src.utils.paths import RESULTS_DIR


def load_predictions(filepath):
    """Wczytuje predykcje i prawdziwe etykiety z pliku CSV."""
    df = pd.read_csv(filepath)
    y_true = df['true_label'].values
    preds = (df['probability'] > 0.5).astype(int)
    return y_true, preds


def main():
    parser = argparse.ArgumentParser(description="Wykonuje test McNemara i eksportuje wyniki do CSV.")
    parser.add_argument("--targets", nargs="+", required=True, 
                        help="Lista celów. Format: 'folder:prefiks:sufiks:zawiera'. "
                             "Puste pola są ignorowane. Przykład samego contains: 'folder:::szukana_fraza'")
    parser.add_argument("--baseline", required=True, 
                        help="Ścieżka do pliku CSV modelu bazowego (względem folderu results)")
    parser.add_argument("--split", default="val", help="Podział danych (val lub test)")
    parser.add_argument("--output", default="mcnemar_raport.csv", help="Nazwa pliku wyjściowego")

    args = parser.parse_args()

    baseline_path = RESULTS_DIR / args.baseline
    if not baseline_path.exists():
        print(f"Błąd: Nie znaleziono pliku baseline'u pod adresem:\n{baseline_path}")
        return

    y_true_base, preds_base = load_predictions(baseline_path)
    baseline_name = baseline_path.stem

    results = []

    for target in args.targets:
        # Rozdzielanie formatu folder:prefiks:sufiks:zawiera
        parts = target.split(":")
        exp = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        suffix = parts[2] if len(parts) > 2 else ""
        contains = parts[3] if len(parts) > 3 else ""

        exp_dir = RESULTS_DIR / args.split / exp / "raw_predictions"
        if not exp_dir.exists():
            print(f"Pomijam: '{exp}' - nie znaleziono folderu {exp_dir}")
            continue

        csv_files = list(exp_dir.glob("*.csv"))

        # Aplikowanie kolejnych warstw filtrowania
        if prefix:
            csv_files = [f for f in csv_files if f.name.startswith(prefix)]
        if suffix:
            csv_files = [f for f in csv_files if f.stem.endswith(suffix)]
        if contains:
            csv_files = [f for f in csv_files if contains in f.stem]

        for csv_file in csv_files:
            if csv_file.resolve() == baseline_path.resolve():
                continue

            y_true_comp, preds_comp = load_predictions(csv_file)

            if len(y_true_base) != len(y_true_comp):
                print(f"  -> Pomijam {csv_file.stem}: niezgodna liczba próbek!")
                continue

            # Obliczenia do Testu McNemara
            both_correct = np.sum((preds_base == y_true_base) & (preds_comp == y_true_comp))
            base_only = np.sum((preds_base == y_true_base) & (preds_comp != y_true_comp))
            comp_only = np.sum((preds_base != y_true_base) & (preds_comp == y_true_comp))
            both_wrong = np.sum((preds_base != y_true_base) & (preds_comp != y_true_comp))

            table = [[both_correct, base_only],
                     [comp_only, both_wrong]]

            p_value = mcnemar(table, exact=False, correction=True).pvalue

            acc_base = np.mean(preds_base == y_true_base)
            acc_comp = np.mean(preds_comp == y_true_comp)

            results.append({
                "Experiment": exp,
                "Compared_Model": csv_file.stem,
                "Baseline": baseline_name,
                "Acc_Model": float(acc_comp),
                "Acc_Baseline": float(acc_base),
                "P_Value": float(p_value),
            })

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by=["Experiment", "Compared_Model"])

        out_dir = RESULTS_DIR / "analysis"
        out_path = out_dir / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True) 

        df_res.to_csv(out_path, index=False)
    else:
        print("\nNie wygenerowano żadnych wyników. Sprawdź ścieżki, nazwy, prefiksy i sufiksy.")


if __name__ == "__main__":
    main()