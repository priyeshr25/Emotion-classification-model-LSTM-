import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Load saved objects
# ----------------------------
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

MAX_LEN = 50   # same max_len used during training

# ----------------------------
# Prediction function
# ----------------------------
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    preds = model.predict(padded)
    index = np.argmax(preds)
    emotion = le.inverse_transform([index])[0]
    confidence = preds[0][index]
    return emotion, confidence

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.title("💬 Emotion Detection App")
st.write("Enter a sentence and the model will predict the emotion.")

user_text = st.text_area("Enter text here:")

if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        emotion, confidence = predict_emotion(user_text)
        st.success(f"**Predicted Emotion:** {emotion}")
        st.info(f"**Confidence:** {confidence:.2f}")