
import streamlit as st
import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import requests


st.set_page_config(page_title="Linguistic DNA",
                   page_icon="💬")


st.title('Linguistic DNA')

file = st.file_uploader('**Upload audio file**', type=['.wav', '.wave', '.flac', '.mp3', '.ogg'])

st.markdown('or **record** yourself')

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
    # display audio data as received on the backend
    st.audio(wav_audio_data, format='audio/wav')

params = {
    "params": ""
}

#if wav_audio_data is not None:
   # params['params'] = wav_audio_data

#if file is not None:
    #params['params'] = file

api_url = 'http://127.0.0.1:8000'
response = requests.get(api_url)


if st.button('**Get results!**'):
    dic = response.json()
    dic['Our']
    #col1, col2, col3, col4 = st.columns(4)
    #col1.metric("British", "50%")
    #col2.metric("American", "10%")
    #col3.metric("Canadian", "30%")
    #col4.metric("Australian", "10%")



#st.session_state
#st.cashing
