import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
import cv2

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Train, Draw, Predict!")

digits = datasets.load_digits()
X = digits.data / 16.0
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_data(show_spinner=True)
def train_model():
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=300,
        alpha=1e-4,
        solver='sgd',
        random_state=1,
        learning_rate_init=0.1
    )
    model.fit(x_train, y_train)
    return model

if "mlp" not in st.session_state:
    st.session_state.mlp = None

if st.button("Train Model"):
    st.session_state.mlp = train_model()
    st.success("Model trained successfully!")

st.header("Draw a Digit Below:")
canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=192,
    height=192,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    st.image(img, caption="Your Drawing", width=128)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img_resized = cv2.resize(img_gray, (8, 8), interpolation=cv2.INTER_AREA)
    img_inverted = 255 - img_resized
    img_normalized = (img_inverted / 255.0) * 16
    input_data = img_normalized.reshape(1, -1)

    if st.session_state.mlp:
        prediction = st.session_state.mlp.predict(input_data)
        st.subheader(f"Predicted Digit: {prediction[0]}")
    else:
        st.info("Train the model first to make predictions.")
