import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
import librosa
import timeit
from json import encoder
from joblib import Parallel, delayed
import json
import base64

import requests
import av
from flask import Flask, render_template, request
from keras.models import load_model
# import numpy as np
# from keras.saving.model_config import model_from_json
# import librosa
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
# !pip install librosa
import librosa
import librosa.display
import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
# import IPython.display as ipd
# from IPython.display import Audio
# import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from googletrans import Translator


# CSS code
css_code = '''
<style>
.container {
  padding: 20px;
  background-color: #fff5e1;
  border-radius: 10px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

h1 {
  color: #333333;
}
.container1 {
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f0f0f0;
      border: 1px solid #ccc;
      border-radius: 5px;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
      float:right;
    
    }

.status {
  margin-top: 20px;
  font-size: 18px;
  font-weight: bold;
}

.upload {
  margin-top: 20px;
}

.button {
  margin-top: 20px;
  padding: 10px 20px;
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.button:hover {
  background-color: #45a049;
}

.warning {
  color: red;
  font-weight: bold;
  margin-top: 20px;
}
.container2 {
    position: absolute;
    right: -100px;
    color: red;
}
.title{
color:white;
}
</style>
'''

# JavaScript code
js_code = '''
<script>
// Add any JavaScript code here if needed
</script>
'''

# HTML code
html_code = f'''
{css_code}

{js_code}
'''
# Function to extract features from the audio file

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
from googletrans import Translator

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
    translator = Translator()
    translated_text = translator.translate(emotion, dest='ar').text
except:
    emotion = ""

    if not emotion:
        st.session_state["run"] = "true"
    else:
        st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

ravdess = "C:/Users/salman/Downloads/archive (6)/audio_speech_actors_01-24/"
ravdess_directory_list = os.listdir(ravdess)
file_emotion = []
file_path = []
# هذا اللوب يقرا المجلدات فقط
for i in ravdess_directory_list:
    # as their are 24 different actors in our previous directory we need to extract files for each actor
    #     print(i)
    actor = os.listdir(ravdess + i)
    for f in actor:  # يقرا الملفات الصوتية التي داخل المجلد
        part = f.split('.')[0].split('-')
        #         print(f)
        #         print(part)
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(ravdess + i + '/' + f)
#
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])

# يتم دمج التعبير والباث في داتا فريم وحدة
ravdess_df = pd.concat([emotion_df, path_df], axis=1)
# changing integers to actual emotions.
ravdess_df.Emotions.replace({1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust',
                             8: 'Surprise'},
                            inplace=True)

data, sr = librosa.load(file_path[0])
Emotions = pd.read_csv('new_emotion.csv')
X = Emotions.iloc[: ,:-1].values
Y = Emotions['Emotions'].values
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
X_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , 1)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def feat_ext(data):
    #Time_domain_features
    # ZCR Persody features or Low level ascoustic features
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    #Frequency_domain_features
    #Spectral and wavelet Features
    #MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr,n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    return result
def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=feat_ext(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,41))
    i_result = scaler.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result
def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=feat_ext(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,41))
    i_result = scaler.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result


# Function to predict sentiment from the audio file
def prediction1(path1):
    with open('cnnlstmmodel.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    # Load the model weights from a .h5 file
    model.load_weights('cnnlstmmodel000.h5')
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    print(y_pred[0][0])
    return y_pred[0][0]

# Render the Streamlit app
st.set_page_config(page_title="Emation", layout="wide")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
        return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('style/sound_motion_wave_19.jpg')
# def load_lottiefile(filepath:str):
#     with open(filepath,"r")as f:
#         return json.load(f)
#
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()
# lottie_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_yr6skceg.json")
# coding =load_lottiefile("style/audio.json")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")
with st.container():
    st.markdown("<h1 style='color:white; background-color:purple; text-align: center;'>Emotion Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 26px;'>Develop a system that analyzes audio clips of a person speaking and incorporate a camera feature that can assess the emotions of the user in real time by uses artificial intelligence to determine the emotions. Based on the detected emotions, the system will provide recommendations for relevant content on YouTube, such as songs or podcasts, that align with those emotions.</p>", unsafe_allow_html=True)


# Add some CSS styles to center the container
st.markdown("""
    <style>
        .stContainer {
            display: flex;
            background-color: #f5f5f5 !important;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)



def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    set_background('C:/Users/LENOVO/Desktop/text_decation/Sensitive-Data-Project/Sensitive Text Data Indicator DATATERA/AI.png')
path =''
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.markdown("<h1 style='color:white; background-color:purple; text-align: center;' >Audio analysis</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 26px;'>Here we can display an audio file and our model will predict the emotion of the person speaking</p>",
            unsafe_allow_html=True)
    with text_column:
        st.subheader("Record your Voise")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key="audio-upload")
        if uploaded_file is not None:
            st.audio(uploaded_file)
            path = f"C:/Users/salman/Downloads/archive (6)/audio_speech_actors_01-24/Actor_01/{uploaded_file.name}"
            predicted_sentiment = prediction1(path)
            st.write(f"Predicted sentiment: {predicted_sentiment}")


path =''
with st.container():
    st.write("---")
    st.write("##")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.markdown("<h1 style='color:white; background-color:purple; text-align: center;' >Face analysis </h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 26px;'>Here we have the camera feature which can detect the user's emotions in real time. Based on the detected emotions, our system will make recommendations for relevant content on YouTube, such as songs or podcasts, that align with those sentiments.</p>",
            unsafe_allow_html=True)

    with text_column:
        st.subheader("Record your face")
        lang = st.text_input("Enter language here", key="language_input")
        singer = st.text_input("Title")

        if lang and singer and st.session_state["run"] != "false":
            webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

        btn = st.button("Recommend me")
        if btn:
            if not emotion:
                st.warning("Please let me capture your emotion first")
                st.session_state["run"] = "true"
            else:
                webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{translated_text}+{singer}")
                np.save("emotion.npy", np.array([""]))
                st.session_state["run"] = "false"





# Create four columns for four team members
col1, col2, col3, col4 = st.columns(4)

# Create a card for the first team member
with col1:
    st.markdown("""
    <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
        <h3>Fahad alluqmani</h3>
        <p>GitHub: <a href='https://github.com/fahadalhrbi91' target='_blank'>click</a></p>
        <p>LinkedIn: <a href='https://www.linkedin.com/in/fahad-alloqmani' target='_blank'>click</a></p>
    </div>
    """, unsafe_allow_html=True)

# Create a card for the second team member
with col2:
    st.markdown("""
    <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
        <h3>Salman Almalki</h3>
        <p>GitHub: <a href='https://github.com/Salman-mlk77' target='_blank'>click</a></p>
        <p>LinkedIn: <a href='https://www.linkedin.com/in/salman-almalki' target='_blank'>click</a></p>
    </div>
    """, unsafe_allow_html=True)

# Create a card for the third team member
with col3:
    st.markdown("""
    <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
        <h3>Fahad Alotaibi</h3>
        <p>GitHub: <a href='https://github.com/Fahadl' target='_blank'>click</a></p>
        <p>LinkedIn: <a href='https://www.linkedin.com/in/fahad-alotaibi-917aba127/' target='_blank'>click</a></p>
    </div>
    """, unsafe_allow_html=True)

# Create a card for the fourth team member
with col4:
    st.markdown("""
    <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
        <h3>Sultan Alharbi</h3>
        <p>GitHub: <a href='https://github.com/SultanAbAlharbi' target='_blank'>click</a></p>
        <p>LinkedIn: <a href='https://www.linkedin.com/in/sultan-alharbi-a6a166201' target='_blank'>click</a></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(html_code, unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key="audio-upload")
# img_path = f"E://web/New folder//ATM//audio_speech_actors_01-24//Actor_01//{uploaded_file.name}"


        # Use the predict_sentiment function to get the predicted sentiment of the uploaded audio file


# Render the combined HTML and JavaScript code
# st.markdown(html_code, unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key="audio-upload")
# img_path = f"E://web/New folder//ATM//audio_speech_actors_01-24//Actor_01//{uploaded_file.name}"
# st.write(img_path)
# if uploaded_file is not None:
#     st.audio(uploaded_file)
# st.write(f"Hello baby{uploaded_file.name}")

# Add any additional Streamlit elements here



# st.markdown("<div style='background-color: fff5e1; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); text-align: center;'><h1>تحليل المشاعر بتفاصيل الوجه</h1></div>", unsafe_allow_html=True)
# st.markdown("<div style='background-color: #fff5e1; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); text-align: center;'><h1>تحليل المشاعر بتفاصيل الوجه</h1></div>", unsafe_allow_html=True)
#
# lang = st.text_input("Language")
# singer = st.text_input("Title")
#
# if lang and singer and st.session_state["run"] != "false":
#     webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)
#
# btn = st.button("Recommend me")
# if btn:
#     if not emotion:
#         webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{translated_text}+{singer}")
#         np.save("emotion.npy", np.array([""]))
#         st.session_state["run"] = "false"
#
#     else:
#         st.warning("Please let me capture your emotion first")
#         st.session_state["run"] = "true"