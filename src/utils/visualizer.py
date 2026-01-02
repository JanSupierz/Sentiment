import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from collections import Counter
from tqdm.auto import tqdm
from IPython.display import display

class ModelVisualizer:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Macierz Pomyłek"):
        """Klasyczna macierz pomyłek w ładnej oprawie."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(title, fontsize=14, pad=15)
        plt.ylabel('Rzeczywista etykieta')
        plt.xlabel('Predykcja modelu')
        plt.show()

    @staticmethod
    def plot_certainty_histogram(preds, probs, y_true, lower=0.3, title="Rozkład Pewności Modelu"):
        """Nowoczesny histogram gęstości (KDE) z podziałem na poprawność."""
        upper = 1 - lower
        correctness = np.where(preds == y_true, "Poprawna", "Błędna")
        
        df = pd.DataFrame({'Probability': probs, 'Result': correctness})
        plt.figure(figsize=(10, 5))
        
        sns.histplot(data=df, x='Probability', hue='Result', bins=40, 
                     multiple="stack", palette={'Poprawna': '#2ecc71', 'Błędna': '#e74c3c'},
                     kde=True, alpha=0.7)
        
        plt.axvline(lower, color='black', linestyle='--', alpha=0.5)
        plt.axvline(upper, color='black', linestyle='--', alpha=0.5)
        plt.title(title, fontsize=14)
        plt.xlabel("Prawdopodobieństwo klasy pozytywnej")
        plt.ylabel("Liczba przypadków")
        sns.despine()
        plt.show()

    @staticmethod
    def plot_top_concepts(sf, concept_units, top_n=15):
        """Wykres Lollipop dla najbardziej znaczących konceptów (Z-score)."""
        df_pos = sf.logodds_per_class[1].sort_values("zscore", ascending=False).head(top_n)
        df_neg = sf.logodds_per_class[0].sort_values("zscore", ascending=False).head(top_n)
        
        df_plot = pd.DataFrame({
            "concept_name": [concept_units[c] for c in df_pos["concept"]] + 
                            [concept_units[c] for c in df_neg["concept"]],
            "score": list(df_pos["zscore"]) + list(-df_neg["zscore"]),
            "sentiment": ["Pozytywny"] * top_n + ["Negatywny"] * top_n
        }).sort_values("score")

        plt.figure(figsize=(12, 8))
        colors = {"Pozytywny": "#2ecc71", "Negatywny": "#e74c3c"}
        
        plt.hlines(y=df_plot["concept_name"], xmin=0, xmax=df_plot["score"], 
                   color=[colors[s] for s in df_plot["sentiment"]], alpha=0.5)
        
        for sentiment, color in colors.items():
            mask = df_plot["sentiment"] == sentiment
            plt.scatter(df_plot.loc[mask, "score"], df_plot.loc[mask, "concept_name"], 
                        color=color, s=100, label=sentiment, edgecolors='white', zorder=3)

        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"Top {top_n} Konceptów Dyskryminatywnych (Z-score)", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sentiment_wordclouds(train, train_map, unique_units_to_train, sf, top_n=4, max_words=50, n_gram_range=(1,3)):

        """
        Generuje siatkę WordCloudów dla topowych pozytywnych i negatywnych
        konceptów sentymentu. Każda jednostka (unit) jest traktowana jako
        nierozerwalna fraza.
        """

        # 1. Grupowanie jednostek według Concept ID
        cluster_to_units = {}

        for item in tqdm(train, desc="Building WordCloud clusters"):
                    from src.utils.loader import DataLoader 
                    row_units = DataLoader.get_ngrams(item['clean_bow'], ngram_range=n_gram_range)
                    
                    for u in row_units:
                        if u in train_map and u in unique_units_to_train:
                            cid = train_map[u]
                            normalized_unit = u.replace(" ", "_")
                            cluster_to_units.setdefault(cid, []).append(normalized_unit)


        # 2. Wybór konceptów na podstawie Z-score
        pos_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=False)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        neg_ids = (
            sf.logodds_per_class[1]
            .sort_values('zscore', ascending=True)
            .head(top_n)['concept']
            .astype(int)
            .tolist()
        )

        # 3. Przygotowanie siatki wykresów
        fig, axes = plt.subplots(2, top_n, figsize=(18, 10))

        def draw_wc(ax, cid, colormap, label):
            # Zliczanie częstości FRAZ (nie słów)
            word_freq = Counter(cluster_to_units.get(cid, []))

            wc = WordCloud(
                background_color='white',
                colormap=colormap,
                max_words=max_words,
                width=400,
                height=300,
                regexp=r"\w+"  # pozwala na podkreślniki
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation='bilinear')

            z_score = sf.logodds_per_class[1].loc[cid, 'zscore']
            ax.set_title(
                f"{label} (ID: {cid})\nZ-score: {z_score:.2f}",
                fontsize=12,
                fontweight='bold'
            )
            ax.axis('off')

        # 4. Rysowanie pozytywnych konceptów
        for i, cid in enumerate(pos_ids):
            draw_wc(axes[0, i], cid, 'Greens', "POZYTYWNY")

        # 5. Rysowanie negatywnych konceptów
        for i, cid in enumerate(neg_ids):
            draw_wc(axes[1, i], cid, 'Reds', "NEGATYWNY")

        plt.suptitle(
            "Analiza Semantyczna Top Konceptów Sentymentu",
            fontsize=20,
            y=1.02
        )
        plt.tight_layout()
        plt.show()


    @staticmethod
    def visualize_concept_wordcloud(concept_units, title="Globalna Mapa Odkrytych Konceptów"):
            """
            Wizualizuje reprezentantów wszystkich klastrów. 
            Każdy koncept jest traktowany jako jedna nierozerwalna fraza.
            """
            # Łączymy frazy używając podkreślników, aby WordCloud traktował 
            # "bad acting" jako jeden token "bad_acting"
            text = " ".join([u.replace(" ", '_') for u in concept_units])
            
            wc = WordCloud(
                width=1200, 
                height=600, 
                background_color="white",
                colormap="tab20", # Kolorowa paleta dla różnorodnych tematów
                max_font_size=100,
                random_state=42,
                # Ten regex pozwala na podkreślniki wewnątrz słów
                regexp=r"\w+" 
            ).generate(text)
            
            plt.figure(figsize=(15, 7.5))
            plt.imshow(wc, interpolation='bilinear')
            plt.title(title, fontsize=18, pad=20, fontweight='bold')
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    @staticmethod
    def get_detailed_certainty_stats(preds, probs, y_true, lower=0.3):
        """
        Zwraca DataFrame z dokładnymi statystykami % dla regionów pewności.
        """
        upper = 1 - lower
    
        # Klasyfikacja pewności
        conditions = [
            (probs < lower),
            (probs > upper),
            ((probs >= lower) & (probs <= upper))
        ]
        choices = ["Certain Negative", "Certain Positive", "Uncertain"]
        
        # KLUCZOWA POPRAWKA: Dodajemy default="Unknown" (musi być string!)
        categories = np.select(conditions, choices, default="Unknown")
                
        df = pd.DataFrame({
            'Category': categories,
            'Correct': preds == y_true
        })
                
        # Agregacja wyników
        stats = df.groupby('Category').agg(
            Liczba_przypadkow=('Correct', 'count'),
            Poprawne=('Correct', 'sum')
        )
                
        # Obliczenia procentowe
        total_samples = len(df)
        stats['Accuracy (%)'] = (stats['Poprawne'] / stats['Liczba_przypadkow'] * 100).round(2)
        stats['Coverage (%)'] = (stats['Liczba_przypadkow'] / total_samples * 100).round(2)
                
        # Sortowanie dla czytelności
        order = ["Certain Positive", "Certain Negative", "Uncertain"]
        return stats.reindex([c for c in order if c in stats.index])
    
    @staticmethod
    def display_extreme_errors(data_list, top_n=5):
        """
        Displays reviews where the model was most confident but incorrect.
        """
        # Filter for errors (creates a list of references, not a copy)
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        
        # Sort by probability (Descending)
        # High prob = Model was very sure it was positive, but it was negative
        # Low prob = Model was very sure it was negative, but it was positive
        errors.sort(key=lambda x: x['prob'], reverse=True)

        print(f"\n--- Top {top_n} 'Confident' Positive Errors (Sure it was 1, actually 0) ---")
        display(pd.DataFrame(errors[:top_n])[['clean_review', 'sentiment', 'pred', 'prob']])
        
        print(f"\n--- Top {top_n} 'Confident' Negative Errors (Sure it was 0, actually 1) ---")
        display(pd.DataFrame(errors[-top_n:])[['clean_review', 'sentiment', 'pred', 'prob']])

    @staticmethod
    def display_uncertain_errors(data_list, top_n=10):
        """
        Displays reviews where the model was most uncertain (probability near 0.5).
        """
        # Filter for errors
        errors = [item for item in data_list if item['sentiment'] != item['pred']]
        
        # Calculate uncertainty score: distance from 0.5
        # We don't need to store this in the dict; we can calculate it during sorting
        errors.sort(key=lambda x: abs(x['prob'] - 0.5))

        print(f"\n--- Top {top_n} Most Uncertain Reviews (Prob near 0.5) ---")
        # Use a temporary DataFrame slice for display
        df_display = pd.DataFrame(errors[:top_n]).copy()
        df_display['uncertainty_score'] = (df_display['prob'] - 0.5).abs()
        
        display(df_display[['clean_review', 'sentiment', 'pred', 'prob', 'uncertainty_score']])