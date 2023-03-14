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



st.set_page_config(page_title="Linguistic DNA",
                   page_icon="💬")

st.sidebar.success("Find your accent 📉")

#st.title(':light-blue-70[Linguistic DNA]:dna:')
st.title('Linguistic DNA 🧬')


file = st.file_uploader(':blue[**Upload audio file**]', type=['wav'])

if file is not None:
    audio_bytes = file.read()
    data = {'wav': audio_bytes}
    #st.audio(make_audio_file(audio_bytes), format='audio/wav') --> if we want to also visualize the audio files that are being uploaded, it is a function from streamlit



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



api_url = 'https://dna-api-roger-hauberr-5yrpl53y3a-ew.a.run.app'


if st.button('**Get 5 classifications !**'):
    t_end = time.time() + 25
    flags = [':uk:',':flag-au:',':flag-us:',':flag-ca:']
    t = st.empty()
    while time.time() < t_end:

        t.markdown(f'''
                {flags[0]}{flags[1]}{flags[2]}{flags[3]}
                ''')
        random.shuffle(flags)
        time.sleep(0.5)
    response = requests.post(f'{api_url}/uploadfile', files=data)
    audio = response.json()

# Add a timer before displaying the plot
    #st.write('Wait for your results ⏳')
    #time.sleep(4)

    # Display a balloons message before showing the plot
    st.balloons()

    # sort the dictionary by values in descending order
    sorted_audio = dict(sorted(audio.items(), key=lambda item: item[1], reverse=True))


    country = list(sorted_audio.keys())
    accents = list(sorted_audio.values())

    total_count = sum(accents)
    percentages = [round((accent / total_count) * 100) for accent in accents]


    # Create a bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=country ,
        y=[accent * 100 for accent in accents],
        marker_color=['#1f77b4' if i==0 else 'indianred' for i in range(len(country))],  # Highlight first bar with red color
        text=percentages,  # Add percentages as text on top of bars
        textposition='auto',  # Automatically position text on top of bars
        texttemplate='%{text}%',
        textfont_size=21,  # Set font size of text to 16# Display text as percentage
    ))
    fig.update_layout(
        xaxis_tickangle=0,
        xaxis_tickfont_size=20
    )

# Display the chart using Streamlit
    st.plotly_chart(fig)
    style ='<p style="font-family:sans-serif; color:indianred; font-size: 40px;"'
    text = f'>You have an {country[0]} Accent 🎉</p>'
    new = f'{style}{text}'
    st.markdown(new, unsafe_allow_html=True)




    # display the metrics in the sorted order horizontally
    #cols = st.columns(len(sorted_audio))
    #for i, (country, value) in enumerate(sorted_audio.items()):
      #  cols[i].metric(label=country, value=str(round(100*value))+'%')





if st.button('**Get binary classifications !**'):
    response = requests.post(f'{api_url}/binary', files=data)
    audio = response.json()

# Add a timer before displaying the plot
    #st.write('Wait for your results ⏳')
    #time.sleep(4)

    # Display a balloons message before showing the plot
    #st.balloons()

    # sort the dictionary by values in descending order
    sorted_audio = dict(sorted(audio.items(), key=lambda item: item[1], reverse=True))


    country = list(sorted_audio.keys())
    accents = list(sorted_audio.values())

    total_count = sum(accents)
    percentages = [round((accent / total_count) * 100) for accent in accents]


    # Create a bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=country ,
        y=[accent * 100 for accent in accents],
        marker_color=['#1f77b4' if i==0 else 'indianred' for i in range(len(country))],  # Highlight first bar with red color
        text=percentages,  # Add percentages as text on top of bars
        textposition='auto',  # Automatically position text on top of bars
        texttemplate='%{text}%',
        textfont_size=21,  # Set font size of text to 16# Display text as percentage
    ))
    fig.update_layout(
        xaxis_tickangle=0,
        xaxis_tickfont_size=20
    )

# Display the chart using Streamlit
    st.plotly_chart(fig)
    style ='<p style="font-family:sans-serif; color:indianred; font-size: 40px;"'
    text = f'>You have an {country[0]} Accent 🎉</p>'
    new = f'{style}{text}'
    st.markdown(new, unsafe_allow_html=True)
