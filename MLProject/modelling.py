import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("credit_card_fraud_preprocessed.csv")

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mlflow.set_experiment("Fraud Detection Basic")
mlflow.sklearn.autolog()

model = RandomForestClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

model.score(
    X_test,
    y_test
)