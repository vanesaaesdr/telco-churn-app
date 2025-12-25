import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.drop(columns=["customerID"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

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

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    return model

model = train_model()

# ---------------------------
# HEADER / JUDUL
# ---------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📊 Telco Customer Churn Prediction</h1>
    <p style="text-align:center; max-width:700px; margin:auto;">
    Aplikasi ini memprediksi kemungkinan pelanggan berhenti (churn) menggunakan model Machine Learning yang telah dilatih.
    </p>
    <br>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# INPUT DATA PELANGGAN
# ---------------------------
st.markdown("### 🧾 Input Data Pelanggan")

# Bungkus dalam container agar form rapi di tengah
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
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0, step=10.0)

# tombol prediksi di bawah form
pred_button = st.button("🔍 Prediksi Churn")

# ---------------------------
# PREDIKSI & OUTPUT
# ---------------------------
if pred_button:
    # DataFrame input disusun sesuai kolom dataset Telco IBM (tanpa customerID dan Churn) [file:21]
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": ["No"],          # default sederhana
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [Contract],
        "PaperlessBilling": ["Yes"],       # default sering Yes
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
    })

    # prediksi churn
    y_pred = model.predict(input_df)[0]
    proba_churn = model.predict_proba(input_df)[0][1]
    proba_pct = proba_churn * 100

    # ---------------------------
    # HASIL PREDIKSI (UI MIRIP GAMBAR)
    # ---------------------------
    st.markdown("### 📌 Hasil Prediksi")

    # kartu hijau / merah
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

    st.write("")  # spasi kecil
    st.write(f"**Probabilitas Churn:** {proba_pct:.2f}%")

    # progress bar probabilitas (warna biru seperti contoh)
    prog_col, _ = st.columns([3, 2])
    with prog_col:
        st.progress(int(proba_pct))

    # ---------------------------
    # PENJELASAN FITUR
    # ---------------------------
    st.markdown("### 📘 Penjelasan Fitur")
    st.markdown(
        """
        - **Tenure**: Lama berlangganan (bulan). Semakin singkat biasanya risiko churn lebih tinggi.  
        - **Monthly Charges**: Tagihan bulanan pelanggan.  
        - **Total Charges**: Total tagihan selama berlangganan.  
        - **Contract**: Kontrak jangka panjang (One year/Two year) biasanya membuat churn lebih kecil.  
        - **Payment Method & Paperless Billing**: Pola pembayaran tertentu sering berkaitan dengan churn.
        """
    )
