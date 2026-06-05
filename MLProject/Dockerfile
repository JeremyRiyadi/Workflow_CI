FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn==1.5.2 mlflow==2.19.0

CMD ["python", "modelling.py"]