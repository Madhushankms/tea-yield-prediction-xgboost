# Tea Yield Prediction System 🍃

A complete full-stack machine learning system for predicting tea leaf yield using environmental and agricultural features. Built for undergraduate AI assignment.

## 🎯 Project Overview

This system predicts tea yield (kg/hectare) based on 9 key features:

- Rainfall (mm)
- Temperature (°C)
- Fertilizer (kg/hectare)
- Soil pH
- Humidity (%)
- Altitude (meters)
- Sunlight Hours (per day)
- Plant Age (years)
- Pruning Frequency (per year)

**Dataset**: 60,000 synthetic records  
**Model**: XGBoost Regressor (NOT deep learning)  
**Problem Type**: Regression

## 🏗️ System Architecture

```
tea_prediction/
├── backend/          # FastAPI backend + ML training
│   ├── app/         # FastAPI application
│   ├── ml/          # ML training scripts
│   ├── models/      # Saved XGBoost model
│   ├── data/        # Dataset CSV
│   └── reports/     # SHAP visualizations
├── frontend/        # Next.js 14 TypeScript frontend
│   ├── app/        # Next.js app router
│   ├── components/ # React components
│   └── lib/        # API utilities
└── README.md       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- pip
- npm or yarn

### 1️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate dataset and train model
python -m ml.data_generation
python -m ml.train

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000  
API docs at: http://localhost:8000/docs

### 2️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

## 📊 ML Model Details

### Training Pipeline

- **Algorithm**: XGBoost Regressor
- **Data Split**: 70% train, 15% validation, 15% test
- **Hyperparameter Tuning**: GridSearchCV with 3-fold CV
- **Feature Scaling**: StandardScaler
- **Evaluation Metrics**: RMSE, MAE, R² Score
- **Explainability**: SHAP summary plots

### Grid Search Parameters

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

### Model Outputs

- `models/xgboost_model.pkl` - Trained model with scaler
- `reports/figures/shap_summary_plot.png` - SHAP explainability
- `reports/figures/feature_importance.png` - Feature importance chart

## 🔌 API Endpoints

### Health Check

```http
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

### Predict Yield

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

Response: {
  "prediction": 2450.5,
  "feature_importance": {
    "Rainfall": 0.15,
    "Temperature": 0.12,
    ...
  }
}
```

### Get Features

```http
GET /features
Response: {list of features with descriptions}
```

## 🎨 Frontend Features

### Components

- **PredictionForm.tsx**: Input form with validation
- **ResultCard.tsx**: Prediction display with insights
- **FeatureChart.tsx**: Interactive bar chart (recharts)

### Technologies

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Axios for API calls
- Recharts for visualization

### UI Features

- ✅ Loading spinner during prediction
- ✅ Error handling with user-friendly messages
- ✅ Responsive design (mobile + desktop)
- ✅ Real-time feature importance visualization
- ✅ Yield category classification
- ✅ Actionable insights and recommendations

## 📁 Key Files

### Backend

- `ml/data_generation.py` - Dataset generation
- `ml/train.py` - XGBoost training pipeline
- `app/main.py` - FastAPI application
- `app/schemas.py` - Pydantic models
- `app/utils.py` - Model loader utility
- `requirements.txt` - Python dependencies

### Frontend

- `app/page.tsx` - Main application page
- `app/layout.tsx` - Root layout
- `components/PredictionForm.tsx` - Input form
- `components/ResultCard.tsx` - Results display
- `components/FeatureChart.tsx` - Feature importance chart
- `lib/api.ts` - API client utilities
- `package.json` - Node dependencies

## 🧪 Testing the System

### Test Backend

```bash
cd backend
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall":2500,"temperature":24,"fertilizer":500,"soil_ph":5.0,"humidity":80,"altitude":1200,"sunlight_hours":6,"plant_age":20,"pruning_frequency":3}'
```

### Test Frontend

1. Open http://localhost:3000
2. Fill in the form with sample values
3. Click "Predict Tea Yield"
4. View prediction and feature importance chart

## 📈 Model Performance

Typical performance metrics (after GridSearchCV):

- **RMSE**: ~150-200 kg/hectare
- **MAE**: ~100-150 kg/hectare
- **R² Score**: ~0.85-0.95

## 🔧 Troubleshooting

### Backend Issues

- **Model not found**: Run `python -m ml.train` first
- **Port 8000 in use**: Change port in uvicorn command
- **Import errors**: Ensure virtual environment is activated

### Frontend Issues

- **Port 3000 in use**: Change port with `npm run dev -- -p 3001`
- **API connection failed**: Ensure backend is running on port 8000
- **Module not found**: Run `npm install` again

## 📝 Code Quality

- ✅ Modular architecture
- ✅ Type hints (Python) and TypeScript
- ✅ Professional academic-level comments
- ✅ Clean code structure
- ✅ Error handling throughout
- ✅ Production-ready design patterns

## 🎓 Academic Requirements Met

- ✅ XGBoost (NOT deep learning)
- ✅ NO Linear Regression, Decision Trees, or k-NN
- ✅ Train/validation/test split
- ✅ GridSearchCV for hyperparameter tuning
- ✅ RMSE, MAE, R² evaluation metrics
- ✅ Model saved as `xgboost_model.pkl`
- ✅ SHAP explainability with plots
- ✅ Modular Python structure
- ✅ FastAPI backend with Pydantic schemas
- ✅ Next.js 14 frontend with TypeScript
- ✅ Professional documentation

## 📦 Dependencies

### Backend (Python)

- numpy, pandas, scikit-learn
- xgboost
- matplotlib, shap
- fastapi, uvicorn
- pydantic

### Frontend (Node.js)

- next, react, react-dom
- typescript
- axios
- recharts
- tailwindcss

## 👨‍💻 Development

### Extend the Model

1. Modify `ml/data_generation.py` for new features
2. Retrain with `python -m ml.train`
3. Update schemas in `app/schemas.py`
4. Update frontend types in `lib/api.ts`

### Customize UI

1. Edit Tailwind theme in `tailwind.config.ts`
2. Modify components in `components/`
3. Update styles in `app/globals.css`

## 📄 License

This project is created for educational purposes (undergraduate AI assignment).

## 🤝 Contributing

This is an academic project. Feel free to fork and adapt for your own assignments.

## ✨ Features Highlight

- 🚀 Production-ready full-stack architecture
- 🧠 XGBoost with GridSearchCV optimization
- 📊 SHAP explainability analysis
- 🎨 Modern, responsive UI with Tailwind CSS
- 🔒 Type-safe code (TypeScript + Python type hints)
- 📈 Real-time feature importance visualization
- ⚡ Fast API responses with proper CORS
- 🎯 60,000-record synthetic dataset
- 🔍 Comprehensive error handling
- 📱 Mobile-responsive design

---

**Built with ❤️ for AI/ML education**
