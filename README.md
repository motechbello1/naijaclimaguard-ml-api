# NaijaClimaGuard ML API

Real-time flood risk predictions for Nigerian locations.

## Stack
- FastAPI + XGBoost
- Real data from Open-Meteo API (NASA GPM IMERG derived)
- Trained on 10,000+ samples across 5 Nigerian flood zones (2018-2023)

## Deploy to Render

1. Push this folder to a new GitHub repo
2. Go to render.com → New → Web Service
3. Connect the repo
4. Build Command: `bash build.sh`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Choose Free plan
7. Deploy

The build step automatically trains the model on real historical data.

## API Endpoints

- `GET /` — Service info
- `GET /health` — Health check
- `GET /v1/risk?latitude=7.73&longitude=6.69` — Get flood risk for a location

## Test Locally

```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

Then visit: http://localhost:8000/v1/risk?latitude=7.73&longitude=6.69
