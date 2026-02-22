# Tea Yield Prediction System - Quick Start Guide

## 🎯 Prerequisites Checklist

Before running setup, ensure you have:

- ✅ Python 3.8 or higher installed
- ✅ Node.js 18 or higher installed
- ✅ pip (Python package manager) installed
- ✅ npm (Node package manager) installed
- ✅ At least 2GB free disk space
- ✅ Internet connection for downloading dependencies

Check versions:

```bash
python --version  # Should be 3.8+
node --version    # Should be 18+
npm --version     # Should be 9+
```

## ⚡ Quick Setup (Automated)

### Windows

```bash
setup.bat
```

### Linux/Mac

```bash
chmod +x setup.sh
./setup.sh
```

This will:

1. Create Python virtual environment
2. Install all Python dependencies
3. Generate 60,000-record dataset
4. Train XGBoost model with GridSearchCV
5. Install all Node.js dependencies

**Setup time**: 5-10 minutes (depending on hardware)

## 🚀 Running the Application

### Option 1: Two Terminals

**Terminal 1 - Backend:**

```bash
cd backend
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd frontend
npm run dev
```

### Option 2: Manual Step-by-Step

#### Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate dataset (60,000 records)
python -m ml.data_generation

# Train model (takes 3-5 minutes)
python -m ml.train

# Start server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🌐 Access Points

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 🧪 Quick Test

### Test Backend

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{ "status": "healthy", "model_loaded": true }
```

### Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"rainfall\":2500,\"temperature\":24,\"fertilizer\":500,\"soil_ph\":5.0,\"humidity\":80,\"altitude\":1200,\"sunlight_hours\":6,\"plant_age\":20,\"pruning_frequency\":3}"
```

### Test Frontend

1. Open http://localhost:3000
2. Fill form with default values
3. Click "Predict Tea Yield"
4. View results and feature importance chart

## 📊 What Gets Created

After setup:

```
tea_prediction/
├── backend/
│   ├── venv/                    # Virtual environment
│   ├── models/
│   │   └── xgboost_model.pkl   # Trained model (~50MB)
│   ├── data/
│   │   └── tea_data.csv        # 60,000 records (~3MB)
│   └── reports/
│       └── figures/
│           ├── shap_summary_plot.png
│           └── feature_importance.png
└── frontend/
    └── node_modules/            # Node.js packages (~200MB)
```

## 🔧 Troubleshooting

### Python version error

```bash
# Install Python 3.8+ from python.org
python --version
```

### Virtual environment activation fails

**Windows (PowerShell):**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### Port already in use

**Backend (change to 8001):**

```bash
uvicorn app.main:app --reload --port 8001
```

**Frontend (change to 3001):**

```bash
npm run dev -- -p 3001
```

Then update `NEXT_PUBLIC_API_URL` in frontend.

### Module not found errors

**Backend:**

```bash
pip install -r requirements.txt --force-reinstall
```

**Frontend:**

```bash
rm -rf node_modules package-lock.json
npm install
```

### Model training too slow

Training is CPU-intensive. Expected times:

- Modern CPU (8+ cores): 2-3 minutes
- Older CPU (4 cores): 5-10 minutes

You can reduce time by:

1. Decreasing GridSearchCV parameter combinations in `ml/train.py`
2. Using fewer cross-validation folds (change `cv=3` to `cv=2`)

### CORS errors in browser

Ensure backend `app/main.py` has correct CORS settings:

```python
allow_origins=["http://localhost:3000"]
```

## 📝 File Structure Overview

```
tea_prediction/
├── backend/              # FastAPI + ML
│   ├── app/             # FastAPI application
│   │   ├── main.py     # API endpoints
│   │   ├── schemas.py  # Pydantic models
│   │   └── utils.py    # Model loader
│   ├── ml/              # ML pipeline
│   │   ├── data_generation.py
│   │   └── train.py
│   └── requirements.txt
├── frontend/            # Next.js + React
│   ├── app/            # Next.js pages
│   ├── components/     # React components
│   ├── lib/            # API utilities
│   └── package.json
├── README.md           # Main documentation
├── setup.bat           # Windows setup
└── setup.sh            # Linux/Mac setup
```

## 🎓 Academic Requirements Met

✅ XGBoost Regressor (not deep learning)  
✅ NO Linear Regression, Decision Trees, k-NN  
✅ 70/15/15 train/val/test split  
✅ GridSearchCV hyperparameter tuning  
✅ RMSE, MAE, R² metrics  
✅ Model saved as xgboost_model.pkl  
✅ SHAP explainability with plots  
✅ FastAPI backend with Pydantic  
✅ Next.js 14 with TypeScript  
✅ Modular, production-ready code

## 💡 Tips for Submission

1. **Screenshots to include:**
   - Frontend UI with prediction
   - Feature importance chart
   - API documentation (/docs)
   - SHAP plots from reports/figures/
   - Terminal showing model training output

2. **Demo flow:**
   - Show data generation
   - Show model training with GridSearchCV
   - Show API health check
   - Show frontend prediction
   - Explain feature importance

3. **Code walkthrough:**
   - Explain XGBoost implementation
   - Show GridSearchCV configuration
   - Demonstrate SHAP analysis
   - Highlight modular structure

## 📞 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review README.md files in backend/ and frontend/
3. Check console/terminal for error messages
4. Verify all prerequisites are installed

## 🎉 Next Steps

After setup:

1. Explore the code structure
2. Read inline comments
3. Modify parameters and retrain
4. Customize the UI
5. Add your own features

---

**Ready to start?** Run `setup.bat` (Windows) or `./setup.sh` (Linux/Mac)!
