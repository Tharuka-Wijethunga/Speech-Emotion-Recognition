import streamlit as st
import librosa
import numpy as np
from joblib import load
from keras.src.saving.saving_api import load_model


# Load the pre-trained model, scaler, and encoder
@st.cache_resource
def load_model_and_components():
    model = load_model("speech_emotion_recognition_model.h5")
    scaler = load("scaler.joblib")
    encoder = load("encoder.joblib")
    return model, scaler, encoder


# Feature extraction pipeline
def extract_features_pipeline(audio_path):
    # Load audio file
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC (including delta and delta-delta)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    delta2_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
    result = np.hstack((result, mfcc_mean, delta_mfcc, delta2_mfcc))

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result



# Streamlit UI
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file to detect the emotion.")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file (wav, mp3, etc.)", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio_file.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio(uploaded_file, format="audio/wav")

        # Load the model, scaler, and encoder
        model, scaler, encoder = load_model_and_components()

        # Preprocess the audio file
        st.write("Extracting features...")
        features = extract_features_pipeline("temp_audio_file.wav")

        # Scale features
        st.write("Scaling features...")
        features_scaled = scaler.transform([features])

        # Reshape for model input
        features_scaled = np.expand_dims(features_scaled, axis=2)

        # Make prediction
        st.write("Predicting emotion...")
        prediction = model.predict(features_scaled)
        predicted_label = encoder.inverse_transform(prediction)

        # Display the result
        st.success(f"The predicted emotion is: **{predicted_label[0]}**")


if __name__ == "__main__":
    main()
