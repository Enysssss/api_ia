FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Générer le dataset synthétique + entraîner les modèles au build
RUN python ml/data/generate_dataset.py && \
    python ml/src/training/train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
