# 🧪 Tests Directory (`tests/`)

This directory houses automated unit and integration tests using **PyTest** to ensure code reliability across the sentiment analysis pipeline.

---

## 📂 Subdirectories

### 1. `tests/unit/`
* **Purpose:** Tests individual components in isolation.
* **Coverage:**
  * `test_preprocessor.py`: Verifies HTML tag stripping, lowercasing, and handling of empty or edge-case strings.
  * `test_vectorizer.py`: Verifies TF-IDF vocabulary generation and feature matrix dimensions.
  * `test_models.py`: Tests forward passes and output probability shapes for PyTorch and Scikit-Learn models.

### 2. `tests/integration/`
* **Purpose:** Tests interactions between multiple system components.
* **Coverage:**
  * `test_pipeline.py`: Tests the full pipeline from raw text input through preprocessor, vectorizer, and prediction.
  * `test_api.py`: Tests FastAPI REST endpoints (`/health`, `/predict`, `/batch_predict`) using `fastapi.testclient.TestClient`.

---

## 🚀 Running the Tests
To execute all tests from the repository root:
```bash
pytest tests/ -v
```

To run a specific test suite:
```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```
