import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smartphone Price Predictor")

st.title("Smartphone Price Tier Predictor")
st.write("Prediksi apakah smartphone termasuk kelas Mahalan (1) atau Murah (0)")

model = joblib.load("best_model.pkl")

st.sidebar.header("Model Performance")
st.sidebar.write("Accuracy > 90%")

def get_user_input():
    ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=4)
    memory = st.number_input("Storage (GB)", min_value=8, max_value=1024, value=64)
    battery = st.number_input("Battery mAh", min_value=1000, max_value=7000, value=4000)
    display = st.number_input("Screen size (inch)", value=6.0)
    brand = st.text_input("Brand", "Samsung")

    return pd.DataFrame([{
        "ram": ram,
        "memory": memory,
        "battery": battery,
        "display": display,
        "brand": brand
    }])

df_input = get_user_input()
st.write(df_input)

if st.button("Predict"):
    pred = model.predict(df_input)[0]
    st.success(f"Hasil Prediksi: {'Expensive (1)' if pred == 1 else 'Cheap (0)'}")
