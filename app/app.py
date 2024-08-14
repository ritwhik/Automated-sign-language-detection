import time 
import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tensorflow as tf
from gtts import gTTS
import io
import pkg_resources
from symspellpy import SymSpell, Verbosity
from textblob import TextBlob
from language_tool_python import LanguageTool
import re
import pyttsx3  
import base64
import threading
#################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import mediapipe as mp


#################################   #################################################################

# Function to preprocess hand landmarks for CNN input
def preprocess_landmarks(landmarks):
    # Convert landmarks to a NumPy array
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # Flatten the array
    landmarks_array = landmarks_array.flatten()
    
    # Normalize the landmarks to be in the range [0, 1]
    landmarks_array = (landmarks_array - np.min(landmarks_array)) / (np.max(landmarks_array) - np.min(landmarks_array))
    
    # Reshape for CNN input
    return landmarks_array.reshape(1, 21, 3, 1)


class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'space', 27: 'del', 28: 'nothing'
}

# Define the mapping of class labels to their corresponding indices
class_label_indices = {value:key for key, value in class_labels.items()}
#class_label_indices

#################################   Functions for prediction  #######################################
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=1
)

gesture_to_text_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'space', 27: 'del', 28: 'nothing'
}

# Set Streamlit page configuration
st.set_page_config(
    page_title="ASL Gesture Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)




class SpellCheckerModule:
    def __init__(self):
        self.spell_check = TextBlob("")
        self.grammar_check = LanguageTool('en-US')

    def correct_spell(self, text):
        simplified_text = re.sub(r'(.)\1+', r'\1', text)
        words = simplified_text.split()
        corrected_words = []
        for word in words:
            corrected_word = str(TextBlob(word).correct())
            corrected_words.append(corrected_word)
        return " ".join(corrected_words)
spellchecker = SpellCheckerModule()


engine = pyttsx3.init()

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_text(inp_txt):
    suggestions = sym_spell.lookup(inp_txt, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"\w+\d",
                                   transfer_casing=True)
    for suggestion in suggestions:
        return suggestion.term


def generate_audio(inp_txt):
    engine.say(inp_txt)
    engine.runAndWait()


def text_to_speech(text):
    if not text.strip():  # Check if the text is empty or contains only whitespace
        st.warning("No text available for speech synthesis.")
        return None
    tts = gTTS(text, lang='en')
    audio_file = io.BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file




def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
def process_frame(frame, model):
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialize the last prediction time if not already done
    if 'last_prediction_time' not in st.session_state:
        st.session_state['last_prediction_time'] = time.time()

    current_time = time.time()

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # hand_landmarks_list = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
            # x_coords, y_coords = zip(*hand_landmarks_list)
            # x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
            # y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

            # hand_img = frame[y_min:y_max, x_min:x_max]
            # hand_img_resized = cv2.resize(hand_img, (224, 224))
            # hand_img_normalized = hand_img_resized / 255.0
            # hand_img_batch = np.expand_dims(hand_img_normalized, axis=0)

            try:
                # new detection model with keypoint-landmark based model

                landmarks_array = preprocess_landmarks(hand_landmarks)
                prediction = model.predict(landmarks_array)
                #print(prediction)
                predicted_class = np.argmax(prediction)
                
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)               
                


                predicted_text = gesture_to_text_mapping.get(predicted_class, 'nothing')
                # Check if enough time has passed since the last prediction----time to readjust gesture
                if current_time - st.session_state['last_prediction_time'] >= 3:  # 3 seconds delay
                    if predicted_text == 'space':
                        st.session_state['text'] += " "
                    elif predicted_text == 'del':
                        st.session_state['text'] = st.session_state['text'][:-1]
                    elif predicted_text != 'nothing' and predicted_text in gesture_to_text_mapping.values():
                        st.session_state['text'] += predicted_text

                    # Update the last prediction time
                    st.session_state['last_prediction_time'] = current_time

                    # Draw on the frame
                # Display the predicted gesture
                cv2.putText(frame, f'Gesture: {predicted_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        # No hand landmarks detected
        if st.session_state['text']:
            #print("session-text: "+ st.session_state['text'])
            correct = spellchecker.correct_spell(st.session_state['text'])
            if(correct=='of'):
                print("session-text: "+ st.session_state['text'])
            if correct and correct!='of':
                st.session_state['store_text'] += '\t' + correct
                threading.Thread(target=generate_audio, args=(correct,),daemon=True).start()
                st.session_state['text'] = " "
            else:   #None value is to be avoided in st.session_state['text'] gives 'of' when processed.. most likely garbage value
                pass
    return frame







def main():
    
    st.subheader("Automated Sign Language Detection")

    if 'text' not in st.session_state:
        st.session_state['text'] = " "
    if 'model' not in st.session_state:
        # st.session_state['model'] = load_model('asl_model_v2_1.h5')
        st.session_state['model'] = load_model('sign_language_model_no_timesteps.h5')
    if 'store_text' not in st.session_state:
        st.session_state['store_text'] = " "
    if 'camera_running' not in st.session_state:
        st.session_state['camera_running'] = True
    if 'last_prediction_time' not in st.session_state:
        st.session_state['last_prediction_time'] = time.time()
    if 'audio_file' not in st.session_state:
        st.session_state['audio_file'] = None    


    col1, col2 = st.columns([1, 1])

    with col1:
        frame_placeholder = st.empty()
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Start Camera"):
                if not st.session_state['camera_running']:
                    st.session_state['camera_running'] = True

        with col4:
            if st.button("Stop Camera"):
                st.session_state['camera_running'] = False
    with col2:
        text_placeholder1 = st.empty()
        text_placeholder = st.empty()

        # if st.button("Play Full Transcript Speech"):
        #     if st.session_state['store_text']:
        #         audio_file = text_to_speech(st.session_state['store_text'])
        #         play_audio_autoplay(audio_file)

        if st.button("Full Transcript Speech"):
            if st.session_state['store_text']:
                audio_file = text_to_speech(st.session_state['store_text'])
                if audio_file:  # Ensure audio_file is not None
                    st.audio(audio_file, format='audio/mp3', autoplay=False)
            else:
                st.warning("No text available for speech synthesis.")
    

    if st.session_state['camera_running']:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Failed to open webcam")
            return

        while st.session_state['camera_running']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            frame = process_frame(frame, st.session_state['model'])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, use_column_width=True)
            
            text_placeholder.write(f"**Transcribed Text:** {st.session_state['text']}")
            text_placeholder1.write(f"**Full Transcripts:** {st.session_state['store_text']}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
