import streamlit as st
import os
import subprocess

st.write("Training model di environment Streamlit Cloud...")
if st.button("Train"):
    result = subprocess.run(
        ["python", "train_best_model_cloud.py"],
        capture_output=True, text=True
    )
    st.code(result.stdout + "\n" + result.stderr)
    if os.path.exists("best_model.pkl"):
        st.success("best_model.pkl berhasil dibuat di Cloud")
