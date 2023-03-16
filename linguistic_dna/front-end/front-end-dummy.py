import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import requests
from scipy.io import wavfile
from scipy.io.wavfile import read
import time
import random
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page
import base64



st.set_page_config(page_title="Linguistic DNA",
                   page_icon="💬",
                   layout= 'centered',
                   initial_sidebar_state='collapsed')


#api_url = 'https://dna-api-roger-hauberr-5yrpl53y3a-ew.a.run.app'




st.markdown("<h1 style='text-align: left; color: black; font-size: 80px;'>Linguistic DNA </h1>", unsafe_allow_html=True)

file_ = open("linguistic-dna-project/ linguistic_dna/front-end/DNA.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align: left; color: black; font-size: 20px;'>Find out about your linguistic 🧬 here!</h1>", unsafe_allow_html=True)

#if st.button('Get your Linguistic 🧬'):



want_to_contribute = st.button("Let's go!")
if want_to_contribute:
    switch_page("Find_your_accent")
