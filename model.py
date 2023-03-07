# import necessary libraries
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# set file paths for CSV file and audio files
csv_path='../raw_data/balanced.csv'
audio_path='../raw_data/target_clips'

# load data from CSV file into a pandas DataFrame
df = pd.read_csv(csv_path)

# delete rows in DataFrame that do not have corresponding audio files
audio_files = os.listdir(audio_path)
df_mini = df.loc[df["path"].isin(audio_files)]
df = df_mini

# shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# add columns to the DataFrame with information about the audio files
df["length"] = filelengths
df["num_words"] = [len(sent) for sent in df["sentence"].str.split()]
df["num_words"].sum()/df["length"].sum()

# keep only the rows with accent values
df_mod = df[~df["accent"].isna()]

# define a function to trim or pad audio files to a specified length
def trim_pad_audio(file, cutoff=4, sr=22050):
    aud, sr = librosa.load(file)  
    if aud.shape[0] < cutoff*sr:
        aud = np.pad(aud, pad_width=(0, (cutoff*sr)-aud.shape[0]))
        return aud
    aud = aud[:cutoff*sr,]
    return aud

# load audio files and trim/pad them to the specified length
aud_ser = [trim_pad_audio(file) for file in "../raw_data/target_clips/" + df_mod["path"]]

# compute the average length of audio files in the DataFrame
df_mod["length"].mean()

# compute MFCC features for the audio files and store them in a numpy array
lst_mfcc = []
for aud in aud_ser:
    lst_mfcc.append(librosa.feature.mfcc(y=aud))
arr_mfcc = np.array(lst_mfcc)

# normalize the MFCC features
arr_mfcc_mmx = np.array((arr_mfcc-np.min(arr_mfcc))/(np.max(arr_mfcc)-np.min(arr_mfcc)))

# split the data into training and test sets
X_train = arr_mfcc_mmx[:1800, :]
X_test = arr_mfcc_mmx[1800:, :]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# convert accent labels to categorical format
y_train = df_mod["accent"].iloc[:1800]
y_test = df_mod["accent"].iloc[1800:]
y_train_cat = pd.get_dummies(y_train)
y_test_cat = pd.get_dummies(y_test)
y_train_cat = np.array(y_train_cat)
y_test_cat = np.array(y_test_cat)

# define a base model for a convolutional neural network
input_shape=(20,173,1)
CNNmodel = Sequential()
CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
CNNmodel.add(layers.MaxPooling2D((2, 2)))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(layers.MaxPooling2D((2, 2)))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(layers.Flatten())
CNNmodel.add(layers.Dense(64, activation='relu'))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Dense(32, activation='relu'))
CNNmodel.add(layers.Dense(5, activation='softmax'))

CNNmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = CNNmodel.fit(X_train, y_train_cat, batch_size=16, validation_split=0.2

# evaluate model on test set                )
metrics = CNNmodel.evaluate(
        x=X_test,
        y=y_test_cat,
        batch_size=16,
        verbose=0,
        # callbacks=None,
        return_dict=True)

metrics['accuracy']


#Alternative Model
#     model = Sequential()
#     model.add(layers.Input(shape=input_shape))
#     model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
#     model.add(layers.BatchNormalization(momentum=0.9))
#     model.add(layers.Dropout(rate=0.1))
#     model.add(layers.Dense(50, activation="relu"))
#     model.add(layers.BatchNormalization(momentum=0.9))  # use momentum=0 to only use statistic of the last seen minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.
#     model.add(layers.Dropout(rate=0.1))
#     model.add(layers.Dense(1, activation="soft_max"))