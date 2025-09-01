FROM python:3.12-slim

WORKDIR /app

RUN pip install uvicorn fastapi torch transformers nltk contractions

COPY main.py .
COPY experiments/deployment/3b_model_scripted.pt experiments/deployment/