# 💻 Source Code Directory (`src/`)

The `src/` directory contains the core, modular Python code for the entire sentiment analysis pipeline. It is structured into specialized sub-packages to follow clean architecture and separation of concerns.

---

## 📂 Sub-package Breakdown

```text
src/
├── data/                    # Ingestion, cleaning, splitting, and PyTorch Dataset class
├── features/                # Vectorizers (TF-IDF), tokenizers, and vocabulary handlers
├── models/                  # Model architectures and wrappers
│   ├── sklearn/             # Scikit-Learn models (Logistic Regression, LinearSVC, Naive Bayes)
│   └── pytorch/             # PyTorch neural models (BiLSTM, Transformers)
├── training/                # Dedicated training loops & scripts for sklearn and pytorch
├── evaluation/              # Performance metrics (Accuracy, F1, ROC-AUC) and report generators
├── deployment/              # FastAPI application, Pydantic schemas, and inference pipelines
└── utils/                   # Shared utilities: structured logger, config parsers, seed setting
```

---

## 🛠️ Module Descriptions

### `src/data/`
* Cleans raw text: strips HTML tags (`<br />`), expands contractions, normalizes punctuation.
* Stratifies and splits data into train/val/test partitions.
* Defines PyTorch `IMDBDataset` with batch tokenization and padding.

### `src/features/`
* Builds and fits TF-IDF / Count vectorizers for classical models.
* Tokenizes text and builds word-to-index mappings for neural networks.

### `src/models/`
* `sklearn/`: Wrapper classes for scikit-learn classifiers with standardized `fit`, `predict`, and `predict_proba` methods.
* `pytorch/`: PyTorch `nn.Module` classes (e.g. BiLSTM, CNN, Transformer classifiers).

### `src/training/`
* Scripts to train models, evaluate on validation sets, apply early stopping, and save the best models to `models/saved_models/`.

### `src/evaluation/`
* Computes confusion matrices, precision, recall, F1-scores, and generates evaluation plots.

### `src/deployment/`
* Hosts the FastAPI service (`app.py`), request/response schemas (`schemas.py`), and inference loader (`predictor.py`).

### `src/utils/`
* Reusable helpers: logger initialization, device selection (`cuda` vs. `cpu`), and YAML config loading.
