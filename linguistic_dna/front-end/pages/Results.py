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

api_url = st.session_state['api_url']
data = st.session_state['data']

t_end = time.time() + 50
flags = [':uk:',':flag-au:',':flag-us:',':flag-ca:',':flag-in:']
t = st.empty()

while time.time() < t_end:
    t.markdown(f'''
        <span style="font-size: 9em">{flags[0]}</span>
        <span style="font-size: 9em">{flags[1]}</span>
        <span style="font-size: 9em">{flags[2]}</span>
        <span style="font-size: 9em">{flags[3]}</span>
        <span style="font-size: 9em">{flags[4]}</span>
    ''', unsafe_allow_html=True)
    random.shuffle(flags)
    time.sleep(0.5)

else:
    t.markdown('')

response = requests.post(f'{api_url}/uploadfile', files=data)
audio = response.json()


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
    marker_color=['coral' if i==0 else 'lightblue' for i in range(len(country))],  # Highlight first bar with red color
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
style ='<p style="font-family:sans-serif; color:coral; font-size: 40px;"'
text = f'>You have an {country[0]} Accent 🎉</p>'
new = f'{style}{text}'
st.markdown(new, unsafe_allow_html=True)
