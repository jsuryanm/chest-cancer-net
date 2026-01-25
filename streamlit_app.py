import streamlit as st 
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Chest Cancer Classification",layout="centered")

st.title("Chest Cancer Classification")

st.write("Upload a CT Scan image")

uploaded_file = st.file_uploader("Upload image",
                                 type=["jpg","jpeg","png"])

if uploaded_file:
    st.image(uploaded_file,caption="Uploaded Image",width=400)

    if st.button("Predict"):
        with st.spinner("Running inference"):
            files = {"file":uploaded_file.getvalue()}

            response = requests.post(API_URL,files=files)

            if response.status_code == 200:
                result =  response.json()[0]

                st.success(f"Prediction:**{result['image']}**")
                st.metric("Confidence",result["confidence"])
            else:
                st.error("Prediction failed")