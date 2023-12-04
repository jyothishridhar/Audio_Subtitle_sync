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

# First code: Extract subtitles from video
video_path = r"C:/OTT_PROJECT/audio_subtitle_synchronization/unsync.mp4"
cap = cv2.VideoCapture(video_path)

subtitles = []  # List to store the extracted subtitles
frame_number = 0  # Variable to track the frame number
millisecond_per_frame = 40  # Assuming each frame is 40 milliseconds
second = 0  # Variable to track the second

while cap.isOpened():
    frame_number += 1  # Increment the frame number at the beginning of the loop

    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Accumulate the subtitles for each frame
    subtitle_text = pytesseract.image_to_string(gray)
    
    if subtitle_text:
        # Format the subtitle timestamps in SRT format
        start_timestamp = (frame_number - 1) * millisecond_per_frame
        end_timestamp = frame_number * millisecond_per_frame

        # Create a dictionary for the subtitle
        subtitle = {
            'frame_number': frame_number,
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'text': subtitle_text
        }

        # Add the subtitle dictionary to the subtitles list
        subtitles.append(subtitle)

        # Check if 800 milliseconds have passed
        if frame_number % 20 == 0:
            second += 1
            print(f"Subtitle for {second} seconds: {subtitle_text}")
            subtitle_text = ""  # Reset subtitle_text for the next 800 milliseconds


# Save the subtitles to an SRT file
srt_file = "C:/OTT_PROJECT/audio_subtitle_synchronization/subtitles_unsync.srt"
with open(srt_file, "w") as file:
    for subtitle in subtitles:
        timestamp_line = f"{subtitle['frame_number']}\n{subtitle['start_time']:0>3} --> {subtitle['end_time']:0>3}\n"
        file.write(timestamp_line + subtitle['text'] + "\n\n")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Load the audio file
audio_file = r"C:/OTT_PROJECT/audio_subtitle_synchronization/unsync.wav"
audio, sr = librosa.load(audio_file, sr=None)

# Extract MFCC features from the audio
mfcc_features = librosa.feature.mfcc(y=audio, sr=sr)
mfcc_features = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)
mfcc_features = np.swapaxes(mfcc_features, 0, 1)

# Prepare the labels
labels = np.array([1] * mfcc_features.shape[0])

# Split the data into training and validation sets
split_ratio = 0.8
split_index = int(split_ratio * mfcc_features.shape[0])
train_features = mfcc_features[:split_index]
train_labels = labels[:split_index]
val_features = mfcc_features[split_index:]
val_labels = labels[split_index:]

# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=train_features.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(train_features, train_labels, batch_size=32, epochs=10, validation_data=(val_features, val_labels))

# Predict voice probabilities for all frames
voice_probabilities = model.predict(mfcc_features)
print('voice_probabilities:', voice_probabilities)

# Load the extracted subtitles from the SRT file
subtitles = []
with open(srt_file, "r") as file:
    subtitle_lines = file.read().split("\n\n")
    for line in subtitle_lines:
        lines = line.split("\n")
        if len(lines) >= 3:
            if " --> " in lines[1]:
                start_time_str, end_time_str = lines[1].split(" --> ")
                start_time = float(start_time_str.replace(",", ".").replace(":", "."))
                end_time = float(end_time_str.replace(",", ".").replace(":", "."))
                text = lines[2]
                frame_number = int(lines[0])  # Get the frame number from the SRT file
                subtitle = {
                    'frame_number': frame_number,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                }
                subtitles.append(subtitle)

# Create a DataFrame from the subtitles list
subtitle_data = pd.DataFrame(subtitles)

# Get the frame numbers corresponding to the original subtitles
original_frame_numbers = subtitle_data['frame_number']

# Print only the original subtitles
for i, subtitle in enumerate(subtitles):
    if i < len(original_frame_numbers):
        original_subtitle = subtitle_data.loc[subtitle_data['frame_number'] == original_frame_numbers[i]].iloc[0]
        print("Original Subtitle:")
        print(f"Frame number: {original_subtitle['frame_number']}, Start time: {original_subtitle['start_time']}, End time: {original_subtitle['end_time']}, Text: {original_subtitle['text']}")
        print()

# Specify the file path for the Excel report
excel_file = 'C:/OTT_PROJECT/audio_subtitle_synchronization/report_unsync_reference.xlsx'

# Save the DataFrame to an Excel file
subtitle_data.to_excel(excel_file, index=False)