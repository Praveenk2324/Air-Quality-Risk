# Air Quality Health Risk Scorer

An end-to-end MLOps and streaming pipeline that predicts health risk tiers based on real-time air quality metrics (PM2.5, NO2, O3). 

This project fetches real-world data from the OpenAQ v3 API, trains a PyTorch Multilayer Perceptron (MLP) model, and serves predictions via both a robust FastAPI REST interface (with SHAP explainability) and an asynchronous Apache Kafka streaming pipeline.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Data Pipeline](#data-pipeline)
- [Model Training & Tuning](#model-training--tuning)
- [Serving & API](#serving--api)
- [Kafka Streaming Pipeline](#kafka-streaming-pipeline)
- [Testing](#testing)

---

## Features
* **Data Ingestion**: Automated retrieval of relevant WHO core pollutants (PM2.5, NO2, O3) via the OpenAQ v3 API.
* **Feature Engineering**: Cyclical time encoding (hour, month) and handling of heavily skewed environmental datasets.
* **Deep Learning Model**: A PyTorch MLP classifier optimized for inference speed using `torch.no_grad()`.
* **Hyperparameter Tuning**: Automated search space optimization using Optuna.
* **RESTful API**: Fast and scalable scoring endpoints built with FastAPI.
* **Explainable AI (XAI)**: Integrated `shap.KernelExplainer` to compute feature attributions on-the-fly.
* **Streaming Architecture**: Fully dockerized Apache Kafka cluster with producer/consumer microservices for continuous scoring.

---

## Project Structure

```text
c:\Projects\Air Quality\
├── data/
│   ├── raw/             # Raw CSVs from OpenAQ
│   └── processed/       # Train/val/test splits and fitted Scaler
├── models/              # Serialized PyTorch models (.pt)
├── src/
│   ├── config.py        # Global configurations & feature definitions
│   ├── data/            # Fetching and preprocessing scripts
│   ├── model/           # PyTorch MLP architecture, training, and tuning
│   └── serving/         # FastAPI app and Kafka consumer/producer
├── tests/               # Pytest unit tests
├── Dockerfile           # Containerizes the FastAPI server & dependencies
├── docker-compose.yml   # Orchestrates Kafka, FastAPI, Producer, and Consumer
└── requirements.txt     # Python dependencies
```

---

## Prerequisites
* **Python 3.11+**
* **Docker & Docker Compose** (for Kafka and containerized API)
* **OpenAQ API Key**: Get a free key at Explore OpenAQ.

---

## Setup & Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd "c:\Projects\Air Quality"
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAQ key:
   ```env
   OPENAQ_API_KEY=your_api_key_here
   ```

---

## Data Pipeline

### 1. Fetching Data
Pull recent PM2.5, NO2, and O3 readings. The default pulls a 25km radius around Bengaluru city centre.
```bash
python -m src.data.fetch --lat 12.9716 --lon 77.5946 --radius 25000
```
*Generates `data/raw/latest.csv`.*

### 2. Preprocessing
Transform raw readings into a wide feature matrix, engineer cyclical time features, assign WHO risk tiers, and create stratified splits.
```bash
python -m src.data.preprocess
```
*Generates train/val/test splits and saves the feature scaler to `data/processed/scaler.pkl`.*

---

## Model Training & Tuning

### Hyperparameter Tuning
Find the best architecture and learning rate using Optuna (uses a short 10-epoch search):
```bash
python -m src.model.tune --trials 30
```

### Model Training
*(Note: Run the full training script with the best parameters found by Optuna. E.g., `python -m src.model.train`)*

The resulting best model should be saved to `models/best_model.pt`.

---

## Serving & API

You can run the FastAPI scoring server locally or via Docker. It exposes endpoints for single predictions (with SHAP explanations) and fast batch processing.

**Run Locally:**
```bash
uvicorn src.serving.api:app --reload
```
API Docs will be available at: http://localhost:8000/docs

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"pm25": 45.2, "no2": 15.1, "o3": 20.5, "hour_of_day": 14, "month": 6, "explain": true}'
```

---

## Kafka Streaming Pipeline

The project includes a Docker Compose setup that provisions an Apache Kafka broker alongside your FastAPI server, a mock sensor producer, and an ML scoring consumer.

**Start the entire cluster:**
```bash
docker-compose up --build -d
```

**What this spins up:**
* `kafka`: Apache Kafka cluster listening on ports 9092 & 29092.
* `fastapi_server`: The REST API exposed on port 8000.
* `sensor_producer`: Python service publishing mock sensor readings.
* `ml_consumer`: Python service consuming data, querying the ML model, and processing streams.

---

## Testing
Run the provided Pytest suite to ensure model shapes, probabilistic outputs, deterministic dropout states, and risk-tier label rules are correct.
```bash
pytest tests/ -v
```