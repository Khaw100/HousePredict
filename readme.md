# 🏠 Housing Price Prediction (End-to-End ML App)

This project is an end-to-end Machine Learning application for predicting housing prices based on structured features from the Ames Housing dataset. It includes data preprocessing, feature engineering, model training with XGBoost, and a web interface for predictions.

## 🚀 Features

- **Data Preprocessing & Feature Engineering**: Handles missing values, categorical encoding, log transformations, and feature selection.
- **XGBoost Regression Model**: Trained on engineered features with hyperparameter tuning.
- **Flask REST API**: Backend service for model inference.
- **Streamlit Frontend**: User-friendly web interface for uploading data and viewing predictions.
- **Docker & Docker Compose**: Containerized deployment for easy setup.
- **MLflow Integration**: Experiment tracking and model versioning.
- **Monitoring**: Logs predictions and model metrics.

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.8+ (if running locally)
- Git

## 🛠 Installation & Setup

### Option 1: Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HomePricesPrediction
   ```

2. Build and run the services:
   ```bash
   docker-compose up --build
   ```

3. Access the application:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:5000

### Option 2: Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r app/backend/requirements.txt
   pip install -r app/frontend/requirements.txt
   ```

2. Run the backend:
   ```bash
   cd app/backend
   python predict.py
   ```

3. Run the frontend (in a new terminal):
   ```bash
   cd app/frontend
   streamlit run streamlit.py
   ```

## 📖 Usage

1. **Upload Data**: Use the Streamlit interface to upload a CSV file with housing features.
2. **Predict Prices**: Click "Predict Housing Prices" to get predictions.
3. **View Results**: See predicted prices alongside input data.
4. **Monitor Metrics**: Check model performance metrics via the API endpoint `/metrics`.

### API Endpoints

- `GET /health`: Health check
- `POST /predict`: Upload CSV and get predictions
- `GET /metrics`: View training and prediction metrics

## 📁 Project Structure

```
HomePricesPrediction/
├── app/
│   ├── backend/          # Flask API
│   └── frontend/         # Streamlit UI
├── data/                 # Sample datasets
├── models/               # Trained models and configs
├── notebooks/            # Jupyter notebooks for development
├── src/                  # Training scripts
├── docker-compose.yml    # Deployment config
└── readme.md
```

## 📚 Model Development Details

To understand how the model was developed, including exploratory data analysis, feature engineering, and training, refer to the notebooks in the `notebooks/` folder:

- `eda.ipynb`: Exploratory Data Analysis
- `feature_engineering.ipynb`: Feature preprocessing and selection
- `modeling.ipynb`: Model training and evaluation

## 📦 Provided Zip File

A zip file containing the complete project codebase, models, and data is provided for easy download and setup. Extract the zip file and follow the installation instructions above.
