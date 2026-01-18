# 🏠 Housing Price Prediction (End-to-End ML App)

This project is an end-to-end Machine Learning application for predicting housing prices based on structured features.
The system includes:

- Feature engineering & preprocessing
- XGBoost regression model
- Flask REST API for inference
- Streamlit frontend for interaction
- Docker & Docker Compose for deployment

---

## 🚧 Known Issue (Important Notice)

⚠️ **Prediction works correctly in local Python execution, but may fail when called from Streamlit UI.**

### ✅ What works
- Model loading ✔
- Feature engineering ✔
- Preprocessing ✔
- Prediction logic ✔
- Reverse log transformation (`expm1`) ✔

All of the above have been verified using a local debug script:

```bash
python debug_predict.py
