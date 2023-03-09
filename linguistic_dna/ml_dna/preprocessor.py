import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# set file paths for CSV file and audio files
csv_path='/Users/frido/linguistic-dna-project/raw_data/balanced.csv'
audio_path='/Users/frido/linguistic-dna-project/raw_data/target_clips/'

# load data from CSV file into a pandas DataFrame
df = pd.read_csv(csv_path)

# delete rows in DataFrame that do not have corresponding audio files
audio_files = os.listdir(audio_path)
df_mini = df.loc[df["path"].isin(audio_files)]
df = df_mini

# shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# add columns to the DataFrame with information about the audio files
filelengths = [librosa.get_duration(path=file) for file in audio_path+df["path"]]
df["length"] = filelengths
df["num_words"] = [len(sent) for sent in df["sentence"].str.split()]
df["num_words"].sum()/df["length"].sum()

# # keep only the rows with accent values
# df_mod = df[~df["accent"].isna()]

# load audio files and trim/pad them to the specified length
aud_ser = [trim_pad_audio(file, cutoff=7) for file in audio_path + df_mod["path"]]

# compute the average length of audio files in the DataFrame
df_mod["length"].mean()

# compute MFCC features for the audio files and store them in a numpy array
lst_mfcc = []
for aud in aud_ser:
    lst_mfcc.append(librosa.feature.mfcc(y=aud))
arr_mfcc = np.array(lst_mfcc)
print('MFCC features computed')

# # normalize the MFCC features
# arr_mfcc_mmx = np.array((arr_mfcc-np.min(arr_mfcc))/(np.max(arr_mfcc)-np.min(arr_mfcc)))

# split the data into training and test sets
X_train = arr_mfcc_mmx[:1800, :]
X_test = arr_mfcc_mmx[1800:, :]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print(X_train[0][:])

# convert accent labels to categorical format
y_train = df_mod["accent"].iloc[:1800]
y_test = df_mod["accent"].iloc[1800:]
y_train_cat = pd.get_dummies(y_train)
y_test_cat = pd.get_dummies(y_test)
y_train_cat = np.array(y_train_cat)
y_test_cat = np.array(y_test_cat)


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
    mfcc = librosa.feature.mfcc(y=aud)

    mfcc_norm = np.array((mfcc-np.min(mfcc))/(np.max(mfcc)-np.min(mfcc)))

    return mfcc_norm

def preprocess(file, cutoff=4, sr=22050):
    """
    Combine both steps: first make audio time series and trim pad and then get the mfcc and normalize.
    """
    aud = trim_pad_audio(file, cutoff=cutoff, sr = sr)

    mfcc_norm = get_norm_mfcc(aud)

    return mfcc_norm
