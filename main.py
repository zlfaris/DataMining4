import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smartphone Price Classifier", layout="centered")

st.title("📱 Smartphone Price Classifier")
st.write("Prediksi apakah smartphone termasuk kategori *Mahal (1)* atau *Murah (0)*")

model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.sidebar.header("Model Info")
summary = pd.read_json("training_summary.json")
st.sidebar.write(summary)

def user_input():
    st.subheader("Input Features")
    data = {}

    # numeric values standardized
    for col in summary["num_columns"][0]:
        data[col] = st.number_input(col, value=0.0)

    # categorical text inputs
    for col in summary["categorical_columns"][0]:
        data[col] = st.text_input(col, value="unknown")

    return pd.DataFrame([data])

user_df = user_input()

if st.button("Predict"):
    prediction = model.predict(user_df)
    st.subheader("Prediction Result")
    st.success(f"Hasil Prediksi: *{int(prediction[0])}* (1=Mahal, 0=Murah)")
    try:
        proba = model.predict_proba(user_df)
        st.write("Confidence:", proba)
    except:
        pass
