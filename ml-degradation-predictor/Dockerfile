FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build artifacts inside the image if they are missing, so the container starts ready to use.
RUN [ -f models/risk_model.joblib ] || python scripts/run_pipeline.py --source auto

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
