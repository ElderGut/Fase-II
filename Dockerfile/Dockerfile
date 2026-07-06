FROM python:3.11-slim

WORKDIR /project

COPY app/requirements.txt app/requirements.txt

RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/ app/
COPY frontend/ frontend/

RUN python app/train_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
