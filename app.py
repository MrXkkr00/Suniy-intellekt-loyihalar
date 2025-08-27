import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import warnings

# Ogohlantirishni o‘chirib qo‘yish
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner")

# Path fix (Windows bilan ishlash uchun)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Transport klassifikatsiya qiluvchi model")

# Modelni yuklash
model = load_learner('./transport_model.pkl')

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type='jpg')
if file:
    st.image(file)

    img = PILImage.create(file)

    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma
    fig = px.bar(x=probs*100, y=model.dls.vocab, orientation='h')
    st.plotly_chart(fig)
