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



st.set_page_config(page_title="Linguistic DNA",
                   page_icon="💬",
                   layout= 'wide',
                   initial_sidebar_state='collapsed')




st.markdown("<h1 style='text-align: left; color: black; font-size: 80px;'>Linguistic DNA 🧬</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left; color: black; font-size: 20px;'>Find out about your linguistic 🧬 here!</h1>", unsafe_allow_html=True)

#if st.button('Get your Linguistic 🧬'):



want_to_contribute = st.button("Find my accent!")
if want_to_contribute:
    switch_page("Find your accent")
