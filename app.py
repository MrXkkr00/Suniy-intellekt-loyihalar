import streamlit as st
from fastai.vision.all import PILImage, load_learner
import pathlib
import plotly.express as px
import warnings
import platform
from PIL import Image


plt = platform.system()
if plt=='Linux':pathlib.WindowsPath = pathlib.PosixPath



st.title("Transport klassifikatsiya qiluvchi model")

# Modelni yuklash
model = load_learner('./transport_model.pkl')

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type=['jpg', 'png'])
if file:
    st.image(file)

    # img = PILImage.create(file)
    img = Image.open(file)
    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma
    fig = px.bar(x=probs*100, y=model.dls.vocab, orientation='h')
    st.plotly_chart(fig)
