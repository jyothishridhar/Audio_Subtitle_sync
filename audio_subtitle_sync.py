import cv2
import pytesseract
import re
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
import tempfile
import os
import base64

# Set Tesseract path using environment variable
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')
print(os.environ.get('TESSERACT_CMD', 'tesseract'))

# Function to download a file from a URL
def download_file(url, dest_path):
    response = requests.get(url)
    with open(dest_path, 'wb') as file:
        file.write(response.content)

# Function for subtitle extraction
def extract_subtitles(video_path):
    cap = cv2.VideoCapture(video_path)
    subtitles = []
    frame_number = 0
    millisecond_per_frame = 40
    second = 0

    while cap.isOpened():
        frame_number += 1
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use tesseract_cmd argument
        subtitle_text = pytesseract.image_to_string(gray, config='--tessdata-dir "./tessdata"')

        if subtitle_text:
            start_timestamp = (frame_number - 1) * millisecond_per_frame
            end_timestamp = frame_number * millisecond_per_frame

            subtitle = {
                'frame_number': frame_number,
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'text': subtitle_text
            }

            subtitles.append(subtitle)

            if frame_number % 20 == 0:
                second += 1
                print(f"Subtitle for {second} seconds: {subtitle_text}")
                subtitle_text = ""

    srt_file = "subtitles_sync.srt"
    with open(srt_file, "w") as file:
        for subtitle in subtitles:
            timestamp_line = f"{subtitle['frame_number']}\n{subtitle['start_time']:0>3} --> {subtitle['end_time']:0>3}\n"
            file.write(timestamp_line + subtitle['text'] + "\n\n")

    return srt_file, subtitles

# Function for audio processing
def process_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=None)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_features = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)
    mfcc_features = np.swapaxes(mfcc_features, 0, 1)
    labels = np.array([1] * mfcc_features.shape[0])

    split_ratio = 0.8
    split_index = int(split_ratio * mfcc_features.shape[0])
    train_features = mfcc_features[:split_index]
    train_labels = labels[:split_index]
    val_features = mfcc_features[split_index:]
    val_labels = labels[split_index:]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=train_features.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.fit(train_features, train_labels, batch_size=32, epochs=10, validation_data=(val_features, val_labels))

    voice_probabilities = model.predict(mfcc_features)
    return voice_probabilities

# Function to run the combined process
def run_combined_process(video_path, audio_file):
    srt_file, subtitles = extract_subtitles(video_path)
    voice_probabilities = process_audio(audio_file)

    subtitle_data = pd.DataFrame(subtitles)
    original_frame_numbers = subtitle_data['frame_number']

    for i, subtitle in enumerate(subtitles):
        if i < len(original_frame_numbers):
            original_subtitle = subtitle_data.loc[subtitle_data['frame_number'] == original_frame_numbers[i]].iloc[0]
            print("Original Subtitle:")
            print(f"Frame number: {original_subtitle['frame_number']}, Start time: {original_subtitle['start_time']}, End time: {original_subtitle['end_time']}, Text: {original_subtitle['text']}")
            print()

    result_df = pd.DataFrame({'Frame Number': subtitle_data['frame_number'], 'Start Time': subtitle_data['start_time'],
                              'End Time': subtitle_data['end_time'], 'Text': subtitle_data['text'],
                              'Voice Probabilities': voice_probabilities.flatten()})
    
    return result_df

# Streamlit app code
st.title("Audio and Subtitle Processing Demo")

# URLs for the video and audio
video_url_sync = "https://github.com/jyothishridhar/Audio_Subtitle_sync/raw/master/referance_video.mp4"
audio_url_sync = "https://github.com/jyothishridhar/Audio_Subtitle_sync/raw/master/referance_audio.wav.wav"

# Temporary download paths
video_path_sync = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
audio_path_sync = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

# Download video and audio
download_file(video_url_sync, video_path_sync)
download_file(audio_url_sync, audio_path_sync)

# Run the combined process
result_df_sync = run_combined_process(video_path_sync, audio_path_sync)

# Display the result on the app
st.success("Process completed! Result:")
st.dataframe(result_df_sync)

# Provide download links for the video and report
st.markdown(f"### Download Video")
st.markdown(f"[Download Video]({video_url_sync})")

st.markdown(f"### Download Report")
csv_report_sync = result_df_sync.to_csv(index=False)
b64_sync = base64.b64encode(csv_report_sync.encode()).decode()
st.markdown(f"[Download Report](data:text/csv;base64,{b64_sync})")

# Clean up temporary files
os.unlink(video_path_sync)
os.unlink(audio_path_sync)
