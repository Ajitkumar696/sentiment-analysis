# 🎬 End-to-End Sentiment Analysis (IMDB Movie Reviews)

A modular, production-grade machine learning repository for classifying sentiment (Positive / Negative) on the IMDB movie reviews dataset.

This repository supports an end-to-end lifecycle—from raw text preprocessing and exploratory data analysis to multi-framework model training (classical **Scikit-Learn** and deep learning with **PyTorch** / **Transformers**), rigorous testing, and production REST API deployment.

---

## 📑 Table of Contents
1. [Project Architecture](#-project-architecture)
2. [Folder Structure & Purpose](#-folder-structure--purpose)
3. [Environment Setup](#-environment-setup)
4. [End-to-End Workflow](#-end-to-end-workflow)
   - [Phase 1: Data Preparation](#phase-1-data-preparation)
   - [Phase 2: Exploratory Data Analysis (EDA)](#phase-2-exploratory-data-analysis-eda)
   - [Phase 3: Feature Engineering](#phase-3-feature-engineering)
   - [Phase 4: Modeling (Scikit-Learn vs. PyTorch)](#phase-4-modeling-scikit-learn-vs-pytorch)
   - [Phase 5: Evaluation & Testing](#phase-5-evaluation--testing)
   - [Phase 6: Serving & Deployment](#phase-6-serving--deployment)
5. [Model Comparison Matrix](#-model-comparison-matrix)
6. [Best Practices](#-best-practices)

---

## 🏛️ Project Architecture

```
                  ┌───────────────────────────────┐
                  │      IMDB Raw Reviews CSV     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Text Preprocessing (src/)  │
                  │   HTML strip, normalize, split│
                  └───────┬───────────────┬───────┘
                          │               │
            ┌─────────────┴─────┐   ┌─────┴─────────────┐
            ▼                   │   │                   ▼
   ┌─────────────────┐          │   │          ┌─────────────────┐
   │ TF-IDF / N-gram │          │   │          │ Tokenizer / Emb │
   └────────┬────────┘          │   │          └────────┬────────┘
            │                   │   │                   │
            ▼                   │   │                   ▼
   ┌─────────────────┐          │   │          ┌─────────────────┐
   │  Scikit-Learn   │          │   │          │  PyTorch Models │
   │ LogReg / LinSVC │          │   │          │ BiLSTM / Transf │
   └────────┬────────┘          │   │          └────────┬────────┘
            │                   │   │                   │
            └─────────────┬─────┘   └─────┬─────────────┘
                          │               │
                          ▼               ▼
                  ┌───────────────────────────────┐
                  │   Evaluation & Unit Tests     │
                  │ Accuracy, F1, ROC, Conf Matrix│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   FastAPI Production Server   │
                  │  /predict, /batch, Dockerized │
                  └───────────────────────────────┘
```

---

## 📁 Folder Structure & Purpose

```text
sentiment-analysis/
├── configs/                     # Central configuration files (YAML) for reproducibility
│   └── README.md
├── data/                        # Local dataset storage (Git-ignored)
│   ├── raw/                     # Original, unaltered IMDB dataset files
│   ├── processed/               # Cleaned reviews and train / val / test splits
│   ├── external/                # External lexicons, sentiment dictionaries, or stopwords
│   └── README.md
├── deployment/                  # Deployment & containerization infrastructure
│   ├── docker/                  # Dockerfile & container runtime definitions
│   └── README.md
├── docs/                        # Project architecture diagrams, design notes & research
│   └── README.md
├── models/                      # Trained model weights & artifacts (Git-ignored)
│   ├── checkpoints/             # Intermediate epoch weights saved during training
│   ├── saved_models/            # Final exported production binaries (.joblib, .pt)
│   └── README.md
├── notebooks/                   # Interactive Jupyter Notebooks for EDA and prototyping
│   └── README.md
├── src/                         # Production Python source package
│   ├── data/                    # Ingestion, cleaning (HTML removal), and split logic
│   ├── features/                # Vectorizers (TF-IDF), tokenizers, and vocabulary indices
│   ├── models/                  # Model definitions
│   │   ├── sklearn/             # Classical ML models (Logistic Regression, LinearSVC, Naive Bayes)
│   │   └── pytorch/             # Deep learning architectures (BiLSTM, Transformers)
│   ├── training/                # Training pipelines, loops, learning rate schedules
│   ├── evaluation/              # Metrics calculation, confusion matrices, and error analysis
│   ├── deployment/              # FastAPI application endpoints and inference schemas
│   ├── utils/                   # Helpers: logger setup, device picker (CUDA/CPU), seed control
│   └── README.md
├── tests/                       # Automated testing suite (PyTest)
│   ├── unit/                    # Unit tests for preprocessors, vectorizers, and models
│   ├── integration/             # Integration tests for training pipeline and REST API
│   └── README.md
├── .gitignore                   # Ignores large datasets, model weights, .venv, caches
├── requirements.txt             # Production and training dependencies
└── README.md                    # Root project documentation
```

---

## ⚡ Environment Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.11, 3.12, or 3.13)
- Git

### 2. Create and Activate Virtual Environment
Using a local virtual environment (`.venv`) is recommended to isolate project dependencies:

**On Windows (PowerShell):**
```powershell
# Create virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS (Bash):**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the pinned `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🔄 End-to-End Workflow

### Phase 1: Data Preparation
1. Obtain the **IMDB Dataset** (50,000 labeled movie reviews).
2. Place the raw file into `data/raw/` (e.g. `data/raw/imdb_reviews.csv`).
3. Use `src/data/` preprocessing modules to:
   - Strip HTML tags (e.g. `<br />` tags commonly found in IMDB reviews).
   - Normalize whitespaces, handle contractions, and remove unwanted artifacts.
   - Stratify and split data into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets.
   - Save clean splits into `data/processed/`.

### Phase 2: Exploratory Data Analysis (EDA)
- Open `notebooks/` to inspect:
  - Class distribution (25,000 positive vs. 25,000 negative reviews).
  - Review length distributions and token counts.
  - Most frequent n-grams in positive vs. negative reviews.

### Phase 3: Feature Engineering (`src/features/`)
- **For Scikit-Learn:** Build TF-IDF n-gram matrices (`ngram_range=(1, 2)`, `sublinear_tf=True`, `max_features=25000`).
- **For PyTorch:** Build vocabulary index mappings, token embeddings, and PyTorch `DataLoader` batches with appropriate padding.

### Phase 4: Modeling (Scikit-Learn vs. PyTorch)
- **Classical ML (`src/models/sklearn/`):**
  - Train Logistic Regression baseline (fast, highly interpretable, ~89-90% accuracy).
  - Train Linear Support Vector Classifier (LinearSVC) with calibration.
- **Deep Learning (`src/models/pytorch/`):**
  - Train Bidirectional LSTM (BiLSTM) with word embeddings.
  - Fine-tune a Transformer model (e.g., `distilbert-base-uncased` or `bert-base-uncased`).

### Phase 5: Evaluation & Testing (`src/evaluation/`, `tests/`)
- Evaluate test set performance using:
  - Accuracy, Precision, Recall, F1-Score.
  - Confusion Matrix & ROC-AUC Curve.
- Run automated tests before deploying:
  ```bash
  pytest tests/
  ```

### Phase 6: Serving & Deployment (`src/deployment/`, `deployment/`)
- Start the local FastAPI inference service:
  ```bash
  uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 --reload
  ```
- Interactive Swagger docs will be accessible at: `http://localhost:8000/docs`
- Package with Docker:
  ```bash
  docker build -f deployment/docker/Dockerfile -t sentiment-api:latest .
  docker run -p 8000:8000 sentiment-api:latest
  ```

---

## 📊 Model Comparison Matrix

| Model Architecture | Framework | Training Time | Expected Accuracy | Primary Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | Scikit-Learn | Fast (~10-30s) | ~89.5% | Extremely fast training & inference, low memory |
| **TF-IDF + LinearSVC** | Scikit-Learn | Fast (~30-60s) | ~90.0% | Strong linear margin separation on text |
| **BiLSTM with Embeddings** | PyTorch | Moderate (~5-15m) | ~88-91% | Captures sequential word context and flow |
| **DistilBERT Fine-Tuned** | Transformers / PyTorch | High (~20-40m GPU) | ~93-94% | State-of-the-art semantic contextual understanding |

---

## 🛡️ Best Practices & Design Principles
- **No Hardcoded Values:** Store learning rates, batch sizes, and filepaths in `configs/`.
- **Reproducibility:** Fix random seeds across NumPy, Python random, and PyTorch.
- **Git Hygiene:** Data files (`data/raw/`, `data/processed/`) and heavy model weights (`models/`) are automatically excluded from Git commits via `.gitignore`.
