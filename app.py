import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = r"C:\Clg\TekWorks\Datasets\DataSVR.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


df = pd.read_csv(DATA_PATH)

# Notebook used: x = df.iloc[:, ::-1], y = df.iloc[:, 0]
X = df.iloc[:, ::-1]
y = df.iloc[:, 0]


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

scaler = StandardScaler()

scaler.fit(X)


st.title("SVR Prediction App")

st.subheader("Enter Feature Values")

user_inputs = []

for column in X.columns:

    min_value = float(X[column].min())
    max_value = float(X[column].max())
    median_value = float(X[column].median())

    value = st.number_input(
        label=column,
        min_value=min_value,
        max_value=max_value,
        value=median_value
    )

    user_inputs.append(value)


if st.button("Predict"):

    # Convert user input into dataframe
    input_df = pd.DataFrame([user_inputs], columns=X.columns)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != input_df.shape[1]:
        st.error(
            f"Feature mismatch: model expects {model.n_features_in_}, "
            f"but got {input_df.shape[1]}."
        )
        st.stop()

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    # Show result
    st.success(f"Prediction: {prediction[0]:.4f}")