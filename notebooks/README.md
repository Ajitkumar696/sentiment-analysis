# 📓 Notebooks Directory (`notebooks/`)

This directory is intended for interactive Jupyter Notebooks (`.ipynb`) used for exploratory data analysis (EDA), visual experimentation, and rapid prototyping.

---

## 📌 Suggested Notebook Workflow

1. **`01_imdb_eda.ipynb`**:
   * Inspect dataset shape, column types, and missing values.
   * Analyze class balance between positive and negative reviews.
   * Review length distributions (character count & word count histograms).
   * Word clouds and most frequent n-grams in positive vs. negative reviews.

2. **`02_model_prototyping.ipynb`**:
   * Quick experiments comparing TF-IDF + Logistic Regression vs. Naive Bayes.
   * Confusion matrix visualizations and sample misclassified reviews.

3. **`03_error_analysis.ipynb`**:
   * In-depth inspection of false positives and false negatives (e.g., sarcasm, negations, subtle movie reviews).

---

## 💡 Guidelines
* **Keep notebooks clean:** Move reusable data cleaning and modeling logic into `src/` as soon as it stabilizes.
* **Imports:** Import functions directly from `src/` (e.g. `from src.data.preprocessor import TextPreprocessor`).
