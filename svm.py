import pickle
import streamlit as st

# Load trained SVM model
model = pickle.load(open("svm.pkl", "rb"))

# App title
st.title("📊 SVM Model Prediction App")
st.write("### by Sobia Khan")

# User inputs
x1 = st.number_input("Enter value for X1", step=0.1)
x2 = st.number_input("Enter value for X2", step=0.1)

# Predict button
if st.button("Predict Class"):
    pred = model.predict([[x1, x2]])[0]
    st.success(f"✅ Predicted Class: {pred}")

