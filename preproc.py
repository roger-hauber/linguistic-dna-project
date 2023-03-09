import pandas as pd
import numpy as np
import librosa

def trim_pad_audio(file, cutoff=4, sr=22050):
    """
    Takes an audio file and gets the audio time series from it, trimming or padding
    it to a specified length in seconds (== cutoff)
    """
    aud, sr = librosa.load(file)

    if aud.shape[0] < cutoff*sr:
        #print("short file pre: ", aud.shape)
        aud = np.pad(aud, pad_width=(0, (cutoff*sr)-aud.shape[0]))
        #print("short file post: ",aud.shape)
        return aud
    #print("long file pre: ",aud.shape)
    aud = aud[:cutoff*sr,]
    #print("long file post: ", aud.shape)
    return aud

def get_norm_mfcc(aud):
    """
    Take an audio time series "aud" and get the mfcc from it and then normalize the resulting array.
    """
    mfcc = librosa.feature.mfcc(y=aud, n_mfcc=128)

    mfcc_norm = np.array((mfcc-np.min(mfcc))/(np.max(mfcc)-np.min(mfcc)))

    return mfcc_norm

def preprocess(file, cutoff=7, sr=22050):
    """
    Combine both steps: first make audio time series and trim pad and then get the mfcc and normalize.
    """
    aud = trim_pad_audio(file, cutoff=cutoff, sr = sr)

    mfcc_norm = get_norm_mfcc(aud)

    return mfcc_norm
