# Taxi Wait Time App

Full-stack example app with a FastAPI backend that loads a Gradient Boosting model (joblib) and a Vite + React frontend.

Folder structure

```
taxi-wait-time-app/
│── backend/
│   ├── main.py
│   ├── models/
│   │   └── taxi_gb_model.joblib  (optional - a fallback model is trained at startup if missing)
│   └── requirements.txt
│
│── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── App.css
```

Quick start


1. Backend (Python)

```powershell
cd backend
python -m pip install -r requirements.txt
# If uvicorn is on your PATH you can run:
uvicorn main:app --reload
# If uvicorn scripts are not on PATH (common for user installs), run with the python -m entrypoint instead:
python -m uvicorn main:app --reload
```

On first run the backend will train and save a small fallback model into `backend/models/taxi_gb_model.joblib` if you don't provide one.

2. Frontend (Node.js)

```powershell
cd frontend
# If you don't have Node.js installed, install it first (examples):
# - winget install --id OpenJS.NodeJS.LTS -e
# - choco install nodejs-lts -y
# or download/install from https://nodejs.org/

npm install
npm run dev
```

Open http://localhost:5173 and the frontend will call the backend at http://127.0.0.1:8000/predict

Notes

- If you already have a real trained model, place it at `backend/models/taxi_gb_model.joblib` and the server will load it.
- The categorical mappings are in `backend/main.py` — ensure they match your model preprocessing.
