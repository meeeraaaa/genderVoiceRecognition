import joblib
import sounddevice as sd
from python_speech_features import mfcc

# Load the trained gender classification model
gender_model_file = 'trainedGenderModel.pkl'
clf = joblib.load(gender_model_file)


def record_audio(duration=5, sample_rate=44100):
    print("Recording... Speak now.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait for recording to complete
    return audio_data


# Add two blank lines here
def classify_gender(audio_data, sample_rate):
    audio_features = mfcc(audio_data, samplerate=sample_rate, numcep=26)
    prediction = clf.predict(audio_features.reshape(1, -1))
    return "male" if prediction[0] == 0 else "female"


def main():
    duration = 5  # Duration of recording in seconds
    sample_rate = 44100  # Sample rate of audio

    audio_data = record_audio(duration, sample_rate)
    gender = classify_gender(audio_data, sample_rate)

    print("Detected gender:", gender)


if __name__ == "__main__":
    main()
