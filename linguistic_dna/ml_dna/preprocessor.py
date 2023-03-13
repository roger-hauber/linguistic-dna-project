import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

## TO BE CHANGED
def set_paths(file_name: str, folder_name: str) -> tuple[str, str]:
    '''
    Set file paths for CSV file and audio files
    '''
    csv_path=f'raw_data/{file_name}'
    audio_path=f'raw_data/{folder_name}'
    return csv_path, audio_path

## TO BE CHANGED
# load data from CSV file into a pandas DataFrame
# df = pd.read_csv(csv_path)


def remove_missing_audios(df: pd.DataFrame, audio_path) -> pd.DataFrame:
    '''
    Deletes rows in DataFrame that do not have corresponding audio files
    '''
    audio_files = os.listdir(audio_path)
    df_mini = df.loc[df["path"].isin(audio_files)]
    return df_mini


def shuffle_rows(df: pd.DataFrame, frac=1) -> pd.DataFrame:
    '''
    Shuffle the rows of the DataFrame
    '''
    df = df.sample(frac=frac).reset_index(drop=True)
    return df


def add_filelength(audio_path, df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add column to the DataFrame with information about the audio file lengths
    '''
    filelengths = [librosa.get_duration(path=file) for file in audio_path+df["path"]]
    df["length"] = filelengths
    return df


# ONE audio file
def trim_pad_audio(file, cutoff=4, sr=22050, drop_first_sec = False):
    """
    Takes an audio file and gets the audio time series from it, trimming or padding
    it to a specified length in seconds (== cutoff)
    """
    aud, sr = librosa.load(file)

    if aud.shape[0] < cutoff*sr:
        #print("short file pre: ", aud.shape)
        aud = np.pad(aud, pad_width=(0, (cutoff*sr)-aud.shape[0]))
        if drop_first_sec == True:
            aud = aud[sr+1:]
        #print("short file post: ",aud.shape)
        return aud
    #print("long file pre: ",aud.shape)
    aud = aud[:cutoff*sr,]
    if drop_first_sec == True:
            aud = aud[sr+1:]
    #print("long file post: ", aud.shape)
    return aud


# ALL audio files
def trim_pad_dataset(audio_path: str,
                     df: pd.DataFrame,
                     cutoff: int = 4,
                     sr: int = 22050,
                     drop_first_sec: bool = False) -> list:
    """
    Takes a path with (multiple) audio files and gets the audio time series from each audio file, trimming or padding
    them to a specified length in seconds (== cutoff)
    """
    aud_ser = [trim_pad_audio(file=file, cutoff=cutoff, sr=sr, drop_first_sec=drop_first_sec)
               for file in audio_path + df["path"]]

    print('✅ audios trimmed and padded')
    return aud_ser


def get_mfcc(aud_ser: list) -> np.array:
    """
    Takes list of trimmed and padded audios and returns acoording mfccs
    """
    lst_mfcc = []

    for aud in aud_ser:
        lst_mfcc.append(librosa.feature.mfcc(y=aud))
    arr_mfcc = np.array(lst_mfcc)

    print('✅ MFCC features computed')
    return arr_mfcc


def mfcc_get_minmax(arr: np.array) -> tuple[float, float]:
    """
    Takes an mfcc array "arr" and calculated its min and max
    """
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    return arr_min, arr_max


def normalizer(arr: np.array, arr_min: float, arr_max: float) -> np.array:
    """
    Takes an mfcc array "arr" and its precalculated min and max and normalizes it
    """
    arr_norm = np.array((arr-arr_min)/(arr_max-arr_min))
    return arr_norm


def adding_dims(arr: np.array, axis=-1) -> np.array:
    """
    Adjusts the input shape before training the model by adding one dimension to an array
    """
    X_expand_dims = np.expand_dims(arr, axis=axis)
    return X_expand_dims


# CAUTION: might not work because we need to save the overall min/max of X_train to adjust X_test
def get_norm_mfcc(aud):
    """
    Take an audio time series "aud" and get the mfcc from it and then normalize the resulting array.
    """
    mfcc = librosa.feature.mfcc(y=aud, n_mfcc=128)

    mfcc_norm = np.array((mfcc-np.min(mfcc))/(np.max(mfcc)-np.min(mfcc)))

    return mfcc_norm

# CAUTION: might not work because we need to save the overall min/max of X_train to adjust X_test
def preprocess(file, cutoff=4, sr=22050):
     """
     Combine both steps: first make audio time series and trim pad and then get the mfcc and normalize.
     """
     aud = trim_pad_audio(file, cutoff=cutoff, sr = sr)

     mfcc_norm = get_norm_mfcc(aud)

     return mfcc_norm


def get_cat_target_array(df: pd.DataFrame) -> np.array:
    '''
    Creates dummy coded target variable (accents)
    '''
    y = df["accent"]
    y_cat = pd.get_dummies(y)
    y_cat = np.array(y_cat)
    return y_cat
