<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=FF9900&center=true&vCenter=true&width=700&lines=Amazon+Item+Price+Predictor;Production-Grade+ML+Pipeline;End-to-End+MLOps+%7C+FastAPI+%7C+MongoDB" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Database-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-FF6B35?style=for-the-badge)](https://dagshub.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **A production-ready, end-to-end Machine Learning system** that predicts Amazon product prices using a fully automated training pipeline, experiment tracking, and a containerized REST API — built with engineering practices that match industry standards.

</div>

---

## About the Project

The **Amazon Item Price Predictor** is not just a model — it's a complete **ML system** engineered from the ground up. It covers the full lifecycle: raw data ingestion from MongoDB, automated feature engineering, model training with experiment tracking via MLflow & DagsHub, and live inference through a FastAPI REST endpoint — all containerized with Docker for repeatable deployment.

This project demonstrates command over the **entire ML stack**: from data pipelines and schema validation all the way through production serving — a skillset that puts it squarely in the top tier of what ML engineers ship.

---

## Key Features

| Feature | Description |
|---|---|
| **MongoDB Data Ingestion** | Pulls raw Amazon product data directly from a cloud MongoDB Atlas collection |
| **Schema Validation** | Automated column-level schema checks before any data enters the pipeline |
| **Train / Test Split** | Configurable, reproducible dataset splitting with tracked artifacts |
| **Feature Engineering** | Custom transformers for NLP-based features (NLTK, emoji processing) + scikit-learn pipelines |
| **Model Training** | Trains regression models with full hyperparameter-aware estimator classes |
| **Experiment Tracking** | MLflow + DagsHub integration for metrics, params, and artifact versioning |
| **Model Evaluation** | Systematic evaluation with configurable acceptance thresholds |
| **FastAPI Inference** | Production-grade `/predict` endpoint — upload a CSV, get predictions back as JSON |
| **Dockerized** | Fully containerized app for zero-config deployment anywhere |
| **Structured Logging** | Custom logger module with consistent, traceable output across all pipeline stages |
| **Custom Exception Handling** | `ProjectError` propagates context-rich errors across all modules |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                       │
│                                                             │
│  MongoDB Atlas ──► Data Ingestion ──► Pre-Processing        │
│                          │                                  |
|                    Schema Validation                        |
|                          |                                  |
│                   Train/Test Split                          │
│                          │                                  │
│               Feature Transformation (NLP + Numeric)        │
│                          │                                  │
│                    Model Training  ◄── Constants/Config     │
│                          │                                  │
│              MLflow / DagsHub Experiment Tracking           │
│                          │                                  │
│                   Model Evaluation                          │
│                          │                                  │
│               Final Model Artifacts Saved                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  /predict   │  ◄── CSV Upload
                    │  /train     │
                    └──────┬──────┘
                           │
                    JSON Response
```

---

## Project Structure

```
AmazonPricePredictor/
│
├── app.py                    # FastAPI application entrypoint
├── Dockerfile                # Container definition
├── requirements.txt          # All dependencies
├── setup.py                  # Package setup
│
├── pipeline/                 # Orchestration
│   └── training_pipeline.py  # End-to-end pipeline runner
│
├── components/               # Pipeline stage implementations
│   ├── data_ingestion.py
|   ├── data_preprocessing.py 
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── model_evaluation.py
│
├── entity/                   # Config & artifact dataclasses
├── constants/                # Training pipeline constants
├── schema/                   # YAML schema definitions
├── utils/
│   ├── main_utils/           # General utilities
│   └── ml_utils/             # ML-specific: estimator, metrics
│
├── logger/                   # Custom structured logger
├── exception/                # ProjectError handler
├── notebooks/                # EDA & experimentation notebooks
└── base_dataset/             # Raw dataset files
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, recommended)
- MongoDB Atlas URI
- DagsHub account (for MLflow tracking)

### 1. Clone & Install

```bash
git clone https://github.com/CR4ZYM4D/AmazonPricePredictor.git
cd AmazonPricePredictor
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/AmazonPricePredictor.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

### 3. Run the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Or Run via Docker

```bash
docker build -t amazon-price-predictor .
docker run -p 8000:8000 --env-file .env amazon-price-predictor
```

---

## API Reference

Once running, the interactive Swagger docs are available at **`http://localhost:8000/docs`**

### `GET /train`
Triggers the full end-to-end training pipeline — ingestion → validation → transformation → training → evaluation.

```bash
curl -X GET "http://localhost:8000/train"
```

### `GET /predict`
Upload a CSV file of Amazon product features; receive JSON predictions for the target price column.

```bash
curl -X GET "http://localhost:8000/predict" \
  -F "file=@products.csv"
```

**Response:**
```json
[
  { "product_name": "...", "category": "...", "price": 499.0 },
  ...
]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Framework** | scikit-learn |
| **NLP** | NLTK, emoji |
| **API** | FastAPI + Uvicorn |
| **Database** | MongoDB Atlas (pymongo) |
| **Experiment Tracking** | MLflow + DagsHub |
| **Serialization** | dill |
| **Config** | PyYAML |
| **Containerization** | Docker |
| **Notebook Environment** | Jupyter / ipykernel |

---

## ML Pipeline Deep Dive

### Data Ingestion
Connects to MongoDB Atlas over TLS, queries the target collection, and exports it as a structured DataFrame — with custom logging at every step.

### Data Pre-Processing
- Text-based feature, quantity and unit extraction via NLTK tokenization and emoji normalization
- Encoding of categorical product metadata

### Data Validation
Validates incoming data against a YAML schema (`schema/`) — column names, data types, and required fields — before any transformation runs.

### Feature Transformation
A scikit-learn `Pipeline` applies:
- Numeric imputation and scaling

### Model Training
Trains a regression estimator wrapped in a `PredictorModel` class. All runs are logged to MLflow via DagsHub — parameters, metrics, and model artifacts are versioned.

### Model Evaluation
The trained model is evaluated on the held-out test set against an acceptance threshold. Only models that meet the bar are persisted to `/final_model/`.

---

## Experiment Tracking

All training runs are tracked on **DagsHub + MLflow**:

- **Metrics:** RMSE, MAE, MAPE, R²
- **Parameters:** Model hyperparameters, transformation settings
- **Artifacts:** Preprocessor pickle, trained model pickle

---

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

If this project helped you, consider leaving a star!

</div>
