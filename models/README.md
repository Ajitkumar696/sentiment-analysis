# 💾 Models Directory (`models/`)

This directory is designated for storing trained model binaries, weights, tokenizers, and checkpoint files.

> **Note:** All model binary files (`*.pt`, `*.pth`, `*.joblib`, `*.bin`, `*.onnx`) are ignored by Git via `.gitignore`.

---

## 📂 Subdirectories

### 1. `models/checkpoints/`
* **Purpose:** Saves intermediate model weights during training (e.g. after each epoch in PyTorch).
* **Format:** `checkpoint_epoch_03.pt`, `best_val_loss.pt`.
* **Use Case:** Allows resuming interrupted training and implementing early stopping.

### 2. `models/saved_models/`
* **Purpose:** Stores the final, production-ready exported model artifacts and vectorizers.
* **Examples:**
  * `tfidf_vectorizer.joblib`: Fitted TF-IDF feature extractor.
  * `sklearn_sentiment_model.joblib`: Trained Logistic Regression or LinearSVC model.
  * `pytorch_bilstm.pt`: Final PyTorch model weights and state dictionary.
  * `vocab.json`: Vocabulary index mapping for tokenized text.
