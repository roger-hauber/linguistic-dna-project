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

st.title('Linguistic DNA 🧬')


file = st.file_uploader(':blue[**Upload audio file**]', type=['wav'])

if file is not None:
    audio_bytes = file.read()
    data = {'wav': audio_bytes}
    #st.audio(make_audio_file(audio_bytes), format='audio/wav') --> if we want to also visualize the audio files that are being uploaded, it is a function from streamlit
    st.session_state['data'] = data


st.markdown(':blue[or **record** yourself]')


def st_audiorec():

    # get parent directory relative to current directory
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # Create an instance of the component: STREAMLIT AUDIO RECORDER
    raw_audio_data = st_audiorec()  # raw_audio_data: stores all the data returned from the streamlit frontend
    wav_bytes = None                # wav_bytes: contains the recorded audio in .WAV format after conversion

    # the frontend returns raw audio data in the form of arraybuffer
    # (this arraybuffer is derived from web-media API WAV-blob data)

    if isinstance(raw_audio_data, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, raw_audio_data = zip(*raw_audio_data['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            raw_audio_data = np.array(raw_audio_data)  # convert to np array
            sorted_ints = raw_audio_data[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            # wav_bytes contains audio data in byte format, ready to be processed further
            wav_bytes = stream.read()

    return wav_bytes

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    data = {'wav': wav_audio_data}

    st.session_state['data'] = data


#api_url = 'https://dna-api-roger-hauberr-new-5yrpl53y3a-ew.a.run.app'
api_url = 'https://dna-api-roger-hauberr-5yrpl53y3a-ew.a.run.app'
st.session_state['api_url'] = api_url


five_class = st.button('**Get Result!**')
if five_class:
    if wav_audio_data is None or file is None:
        st.write("Please record or upload first!")
    if wav_audio_data is not None:
        switch_page('Results')
    if file is not None:
        switch_page('Results')
