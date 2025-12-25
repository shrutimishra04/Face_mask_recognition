import os
import io
from PIL import Image
import numpy as np
import streamlit as st

try:
    import tensorflow as tf
except Exception:
    tf = None

from src.utils.visualize_clean import decode_predictions, draw_boxes_on_image, CLASS_NAMES


MODEL_DIR = os.path.join(os.getcwd(), 'outputs', 'saved_model')
H5_PATH = os.path.join(os.getcwd(), 'outputs', 'best.h5')


@st.cache_resource
def load_model():
    if tf is None:
        return None
    if os.path.isdir(MODEL_DIR):
        try:
            return tf.saved_model.load(MODEL_DIR)
        except Exception:
            pass
    if os.path.exists(H5_PATH):
        try:
            return tf.keras.models.load_model(H5_PATH, compile=False)
        except Exception:
            pass
    return None


def preprocess(img: Image.Image, size=416):
    img = img.convert('RGB')
    img_resized = img.resize((size, size))
    arr = np.array(img_resized).astype('float32') / 255.0
    return arr


def run_inference(model, img_arr):
    inp = np.expand_dims(img_arr, axis=0)
    if model is None:
        return None
    try:
        out = model(inp)
    except TypeError:
        out = model(inp, training=False)
    if isinstance(out, dict):
        out = list(out.values())[0]
    out_np = out.numpy() if hasattr(out, 'numpy') else np.array(out)
    return out_np[0]


def main():
    st.title('Face Mask Detector')

    st.sidebar.header('Settings')
    conf_thresh = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.3)

    uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    st.write('Model directory: ', MODEL_DIR)

    model = load_model()
    if model is None:
        st.warning('Model not found or TensorFlow not available. Place a SavedModel in outputs/saved_model or best.h5 in outputs/.')

    if uploaded is not None:
        img = Image.open(io.BytesIO(uploaded.read()))
        st.image(img, caption='Original image', use_column_width=True)

        img_arr = preprocess(img)
        pred = run_inference(model, img_arr)
        if pred is None:
            st.error('No model available to run inference.')
            return
        boxes = decode_predictions(pred, image_size=416, conf_thresh=conf_thresh)

        vis = draw_boxes_on_image(img_arr, boxes)
        st.image(vis, caption='Predictions (resized to 416x416)', use_column_width=True)

        if boxes:
            st.markdown('**Detections**')
            for i, b in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = b
                st.write(f'{i+1}. {CLASS_NAMES[int(cls)]} — conf={conf:.2f} box=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]')


if __name__ == '__main__':
    main()
