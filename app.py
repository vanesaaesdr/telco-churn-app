import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------
# SETTING HALAMAN
# ---------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="wide"
)

# ---------------------------
# TRAIN MODEL DI AWAL (CACHED)
# ---------------------------
@st.cache_resource
def train_model():
    # 1. Load dataset Telco dari file CSV (harus ada di repo)
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 2. Perbaiki TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Target biner
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 4. Drop ID
    df = df.drop(columns=["customerID"])

    # 5. Pisahkan X, y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 6. Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Definisikan fitur numerik & kategorikal
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
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # 10. Train model
    model.fit(X_train, y_train)

    # Kembalikan model yang sudah terlatih
    return model, numeric_features, categorical_features

model, numeric_features, categorical_features = train_model()

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📊 Telco Customer Churn Prediction</h1>
    <p style="text-align:center; max-width:700px; margin:auto;">
    Aplikasi ini memprediksi kemungkinan pelanggan berhenti (churn) menggunakan model Machine Learning.
    </p>
    <br>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# INPUT FORM (UI mirip screenshot)
# ---------------------------
st.markdown("### 🧾 Input Data Pelanggan")

with st.container():
    col_left, col_right, col_empty = st.columns([2.5, 2.5, 3])

    with col_left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=72, value=10, step=1)
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

    with col_right:
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=10.0)

pred_button = st.button("🔍 Prediksi Churn")

# ---------------------------
# PREDIKSI
# ---------------------------
if pred_button:
    # Susun input user sebagai DataFrame dengan kolom sama seperti training
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
    })

    # Prediksi dengan model (pipeline)
    y_pred = model.predict(input_df)[0]
    proba_churn = model.predict_proba(input_df)[0][1]
    proba_pct = proba_churn * 100

    # Hasil Prediksi
    st.markdown("### 📌 Hasil Prediksi")

    if y_pred == 1:
        st.markdown(
            "<div style='background-color:#7f1d1d; padding:12px; border-radius:6px; color:white;'>"
            "🔴 Pelanggan diprediksi <b>CHURN</b>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background-color:#166534; padding:12px; border-radius:6px; color:white;'>"
            "🟢 Pelanggan diprediksi <b>TIDAK CHURN</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write(f"**Probabilitas Churn:** {proba_pct:.2f}%")
    st.progress(int(proba_pct))

    st.markdown("### 📘 Penjelasan Fitur")
    st.markdown(
        """
        - **Tenure**: Lama berlangganan; makin pendek biasanya risiko churn lebih tinggi.  
        - **Monthly Charges**: Tagihan bulanan pelanggan.  
        - **Total Charges**: Total tagihan sejak berlangganan.  
        - **Contract**: Kontrak bulanan cenderung lebih rawan churn.  
        - **Payment Method & Paperless Billing**: Pola pembayaran juga memengaruhi churn.
        """
    )
