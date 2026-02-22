# Tea Yield Prediction System

ML system for predicting tea yield using XGBoost (FastAPI + Next.js)

## Prerequisites

- Python 3.8+
- Node.js 18+

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Train model (if not already trained)
python -m ml.train

# Start backend server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend runs at:** http://localhost:8000

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start frontend server
npm run dev
```

**Frontend runs at:** http://localhost:3000

---

## 📝 Quick Commands

### Train Model

```bash
cd backend
python -m ml.train
```

### Run Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### Run Frontend

```bash
cd frontend
npm run dev
```

---

## 🧪 Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall":150,"temperature":25,"fertilizer":400}'
```

**Built for AI/ML Assignment**
