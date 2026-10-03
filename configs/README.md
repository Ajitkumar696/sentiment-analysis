# ⚙️ Configurations Directory (`configs/`)

This directory contains configuration files (YAML or JSON) for managing project hyperparameters, dataset paths, and experiment settings without modifying Python code.

---

## 📄 Recommended Configuration Files

1. **`base_config.yaml`**:
   * Global settings: Random seeds, split ratios (e.g. 0.8 / 0.1 / 0.1), file paths to `data/` and `models/`.
   * Label mappings: `{"negative": 0, "positive": 1}`.

2. **`sklearn_config.yaml`**:
   * Feature vectorizer settings: `ngram_range`, `max_features`, `min_df`, `sublinear_tf`.
   * Model parameters: Regularization `C`, solvers, maximum iterations for Logistic Regression / LinearSVC.

3. **`pytorch_config.yaml`**:
   * Neural network architecture: `vocab_size`, `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`.
   * Training parameters: `batch_size`, `learning_rate`, `epochs`, `optimizer` (`AdamW`), `early_stopping_patience`.
   * Pretrained transformer name: `distilbert-base-uncased`.

---

## 💡 Best Practices
* **Never hardcode numbers in Python files:** Always pass parameters through a config file.
* **Experiment Tracking:** Save a copy of the config used with each model checkpoint for 100% reproducibility.
