# Tea Yield Prediction - Backend

FastAPI backend with XGBoost machine learning model for tea yield prediction.

## 📁 Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # Model loader utility
├── ml/
│   ├── __init__.py
│   ├── data_generation.py  # Dataset generation
│   └── train.py            # XGBoost training pipeline
├── models/
│   └── xgboost_model.pkl   # Trained model (generated)
├── data/
│   └── tea_data.csv        # Dataset (generated)
├── reports/
│   └── figures/            # SHAP plots (generated)
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Dataset

```bash
python -m ml.data_generation
```

This creates:

- `data/tea_data.csv` - 60,000 records with 9 features + target

### 4. Train XGBoost Model

```bash
python -m ml.train
```

This performs:

- Train/validation/test split (70/15/15)
- GridSearchCV hyperparameter tuning
- Model evaluation (RMSE, MAE, R²)
- SHAP explainability analysis
- Saves model to `models/xgboost_model.pkl`
- Generates plots in `reports/figures/`

Expected output:

```
Training pipeline...
GridSearchCV completed!
Best parameters: {...}
Test Set Performance:
  RMSE: ~180 kg/hectare
  MAE: ~120 kg/hectare
  R² Score: ~0.90
Model saved to models/xgboost_model.pkl
```

### 5. Start FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:

```bash
python -m app.main
```

Server will start at: http://localhost:8000  
API documentation: http://localhost:8000/docs

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Tea Yield

```http
POST /predict
Content-Type: application/json

{
  "rainfall": 2500.0,
  "temperature": 24.0,
  "fertilizer": 500.0,
  "soil_ph": 5.0,
  "humidity": 80.0,
  "altitude": 1200.0,
  "sunlight_hours": 6.0,
  "plant_age": 20.0,
  "pruning_frequency": 3
}
```

Response:

```json
{
  "prediction": 2450.5,
  "feature_importance": {
    "Rainfall": 0.15,
    "Temperature": 0.12,
    "Fertilizer": 0.18,
    "Soil_pH": 0.08,
    "Humidity": 0.1,
    "Altitude": 0.14,
    "Sunlight_Hours": 0.11,
    "Plant_Age": 0.07,
    "Pruning_Frequency": 0.05
  }
}
```

### Get Features

```http
GET /features
```

Returns list of required features and descriptions.

## 🧪 Testing

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall":2500,"temperature":24,"fertilizer":500,"soil_ph":5.0,"humidity":80,"altitude":1200,"sunlight_hours":6,"plant_age":20,"pruning_frequency":3}'
```

### Test with Python

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Prediction
data = {
    "rainfall": 2500.0,
    "temperature": 24.0,
    "fertilizer": 500.0,
    "soil_ph": 5.0,
    "humidity": 80.0,
    "altitude": 1200.0,
    "sunlight_hours": 6.0,
    "plant_age": 20.0,
    "pruning_frequency": 3
}
response = requests.post('http://localhost:8000/predict', json=data)
print(response.json())
```

## 📊 ML Model Details

### Features (9 total)

1. **Rainfall** (mm): 1500-4000
2. **Temperature** (°C): 18-32
3. **Fertilizer** (kg/hectare): 200-800
4. **Soil_pH**: 4.0-6.0
5. **Humidity** (%): 60-95
6. **Altitude** (meters): 500-2000
7. **Sunlight_Hours** (per day): 3-9
8. **Plant_Age** (years): 5-40
9. **Pruning_Frequency** (per year): 1-4

### Target Variable

- **Yield** (kg/hectare): 1000-4000

### Training Configuration

- **Algorithm**: XGBoost Regressor
- **Objective**: reg:squarederror
- **Train**: 42,000 samples (70%)
- **Validation**: 9,000 samples (15%)
- **Test**: 9,000 samples (15%)
- **Hyperparameter Tuning**: GridSearchCV (3-fold CV)
- **Feature Scaling**: StandardScaler

### GridSearchCV Parameters

```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
```

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

### Explainability

- **SHAP Summary Plot**: Shows feature importance
- **Feature Importance Chart**: XGBoost built-in scores
- Saved to `reports/figures/`

## 🛠️ Development

### Add New Features

1. Update `ml/data_generation.py`
2. Retrain model with `python -m ml.train`
3. Update `app/schemas.py` PredictionInput
4. Update `app/utils.py` feature mapping

### Modify Model Parameters

Edit `ml/train.py`:

- Adjust `param_grid` for GridSearchCV
- Change train/val/test split ratios
- Add new evaluation metrics

## 📦 Dependencies

```
numpy==1.24.3          # Numerical computing
pandas==2.0.3          # Data manipulation
scikit-learn==1.3.0    # ML utilities
xgboost==1.7.6         # XGBoost model
matplotlib==3.7.2      # Plotting
shap==0.42.1           # Explainability
fastapi==0.103.1       # Web framework
uvicorn==0.23.2        # ASGI server
pydantic==2.3.0        # Data validation
```

## 🔧 Troubleshooting

### Model not found

```bash
python -m ml.train
```

### Import errors

Ensure virtual environment is activated:

```bash
venv\Scripts\activate  # Windows
```

### Port already in use

```bash
uvicorn app.main:app --reload --port 8001
```

### CORS issues

Update `app/main.py` allow_origins:

```python
allow_origins=["http://localhost:3000", "http://localhost:3001"]
```

## 📈 Expected Performance

- **RMSE**: 150-200 kg/hectare
- **MAE**: 100-150 kg/hectare
- **R² Score**: 0.85-0.95
- **Training Time**: 2-5 minutes (depending on hardware)
- **Prediction Time**: <50ms per request

## 🎯 Academic Requirements

✅ XGBoost Regressor (NOT deep learning)  
✅ Train/validation/test split  
✅ GridSearchCV hyperparameter tuning  
✅ RMSE, MAE, R² metrics  
✅ Model saved as xgboost_model.pkl  
✅ SHAP explainability  
✅ Modular structure  
✅ Professional comments

---

**Part of the Tea Yield Prediction Full-Stack System**
