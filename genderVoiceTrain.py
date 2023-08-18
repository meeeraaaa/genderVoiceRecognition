# Import all modules
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from python_speech_features import mfcc
from time import time
import joblib
from tqdm import tqdm
from pydub import AudioSegment
import librosa

# Load the csv file into data frame
df = pd.read_csv(r'C:\Users\LENOVO\Downloads\projects\voices\cv-valid-train.csv')
# Create two new data frames
df_male = df[df['gender'] == 'male']
df_female = df[df['gender'] == 'female']

# Take only 300 male and 300 female data
df_male = df_male[:300]
df_female = df_female[:300]

# Define the audio path
# Define the audio path
TRAIN_PATH = r'C:\Users\LENOVO\Downloads\projects\voices\cv-valid-train\\'


# The function to convert mp3 to wav


def convert_to_wav(data_frame, m_f, path=TRAIN_PATH):
    for file in tqdm(data_frame['filename']):
        sound = AudioSegment.from_mp3(path + file)

        if m_f == 'male':
            sound.export('male-' + file.split('/')[-1].split('.')[0] + '.wav', format='wav')
        elif m_f == 'female':
            sound.export('female-' + file.split('/')[-1].split('.')[0] + '.wav', format='wav')

    return


# How to use the convert_to_wav() function
convert_to_wav(df_male, m_f='male', path=r'C:\Users\LENOVO\Downloads\projects\voices\cv-valid-train/')
convert_to_wav(df_female, m_f='female', path=r'C:\Users\LENOVO\Downloads\projects\voices\cv-valid-train/')


# Define a function to load the raw audio files
def load_audio(audio_files):
    # Allocate empty lists for male and female voices in this scope
    male_voices_list = []
    female_voices_list = []

    for file in tqdm(audio_files):
        if file.split('-')[0] == 'male':
            male_voices_list.append(librosa.load(file))
        elif file.split('-')[0] == 'female':
            female_voices_list.append(librosa.load(file))

    # Convert the lists into Numpy arrays
    m_voices = np.array(male_voices_list)
    f_voices = np.array(female_voices_list)

    return m_voices, f_voices


# How to use load_audio() function
male_voices, female_voices = load_audio(os.listdir())


# The function to extract audio features
def extract_features(audio_data):
    audio_waves = audio_data[:, 0]
    samplerate = audio_data[:, 1][1]

    features = []
    for audio_wave in tqdm(audio_waves):
        features.append(mfcc(audio_wave, samplerate=samplerate, numcep=26))

    features = np.array(features)
    return features


# Use the extract_features() function
male_features = extract_features(male_voices)
female_features = extract_features(female_voices)


# The function used to concatenate all audio features forming a long 2-dimensional array
def concatenate_features(audio_features):
    concatenated = audio_features[0]
    for audio_feature in tqdm(audio_features):
        concatenated = np.vstack((concatenated, audio_feature))

    return concatenated


# How the function is used
male_concatenated = concatenate_features(male_features)
female_concatenated = concatenate_features(female_features)

# Concatenate male voices and female voices
X = np.vstack((male_concatenated, female_concatenated))

# Create labels
y = np.append([0] * len(male_concatenated), [1] * len(female_concatenated))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Initialize SVM model
clf = SVC(kernel='rbf')

# Train the model
start = time()
clf.fit(X_train, y_train)
print("Training time:", time() - start, "seconds")

# Compute the accuracy score on the train data
train_accuracy = clf.score(X_train, y_train)
print("Train accuracy:", train_accuracy)

# Compute the accuracy score on the test data
test_accuracy = clf.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

# Save the trained model
gender_model_file = 'trainedGenderModel.pkl'
joblib.dump(clf, gender_model_file)
