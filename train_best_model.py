import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 1. Load dataset Telco
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Perbaiki TotalCharges (string kosong -> NaN -> median)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3. Target biner
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Drop ID
df = df.drop(columns=["customerID"])

# 5. Pisahkan X, y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6. Bagi train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Tentukan fitur numerik & kategorikal
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# 8. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 9. Model Logistic Regression
best_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 10. Train
best_model.fit(X_train, y_train)

# 11. Evaluasi cepat
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1-score best_model:", round(f1, 4))

# 12. Simpan model
joblib.dump(best_model, "best_model.pkl")
print("Best model disimpan sebagai best_model.pkl")