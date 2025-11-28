# IMDb Sentiment Classification

This project implements sentiment classification on IMDb movie reviews using multiple models, including **LSTM**, **Linear SVM**, **RBF SVM**, and **Logistic Regression**. It also supports an **ensemble predictor** to combine model predictions and analyze uncertainty.

---

## Features

* Train and evaluate multiple classifiers.
* Ensemble predictions with threshold-based uncertainty handling.
* Visualizations:

  * Confusion matrix
  * Certainty vs correctness plots
  * Unsure and misclassified examples in Jupyter notebooks.
* Flexible preprocessing:

  * Tokenization by **words** or **sentences**
  * Text cleaning with contraction expansion, HTML removal, and lowercasing.
* Optional deduplication to avoid train/test leakage.

---

## Installation

```
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Loading IMDb Dataset

```python
from data_loader import load_reviews

# Load and preprocess data
train_df, val_df, test_df = load_reviews(unit_type="sentence")
```

* `unit_type`: `"sentence"` or `"word"` — controls how text is split.
* Data is deduplicated by default to avoid overlapping reviews between train and test.

---

### Running Experiments

```python
from experiments import run_experiment

results = run_experiment(
    X_train_sets, y_train,
    X_test_sets, y_test,
    texts=test_df['review'],
    model_type="linear-svm"
)
```

* Supported models: `"lstm"`, `"linear-svm"`, `"rbf-svm"`, `"lr"`.
* Outputs are displayed in **Jupyter-friendly tables and plots**.

---

### Ensemble Predictions

* Combines `"all"`, `"pos"`, `"neg"`, `"important"` models.
* Produces:

  * Ensemble accuracy
  * Unsure predictions
  * Misclassified predictions
* Visualizations help detect ambiguous or mislabeled reviews.

---

## Notes on Dataset Quality

* IMDb dataset labels are **based on star ratings**, not pure text sentiment.
* Some reviews labeled as positive may contain mostly negative content (and vice versa).
* The ensemble’s uncertain predictions can help identify these noisy examples.

---

## Visualization Examples

* **Confusion matrix:** see class-wise accuracy.
* **Certainty analysis:** probability vs correctness.
* **Unsure/misclassified examples:** inspect raw text with model probabilities.

---

## Notes on Dataset Quality

* IMDb dataset labels are **based on star ratings**, not pure text sentiment.
* Some reviews labeled as positive may contain mostly negative content (and vice versa).
* The ensemble’s uncertain predictions can help identify these noisy examples.
* **Example of mislabeled review**:

  > "This low-budget erotic thriller that has some good points, but a lot more bad one... As plots go for this type of genre, not too bad. The script is okay, and the story makes enough sense for someone up at 2 AM watching this not to notice too many plot holes. But everything else in the film seems cheap. The lead actors aren't that bad, but pretty much all the supporting ones are unbelievably bad (one girl seems like she is drunk and/or high). The cinematography is badly lit, with everything looking grainy and ugly. The sound is so terrible that you can barely hear what people are saying. The worst thing in this movie is..."  

  *Labeled as positive, but content is mostly negative.*

