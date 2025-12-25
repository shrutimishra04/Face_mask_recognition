"""Run inference with a SavedModel and OpenCV webcam input.
Usage: python src/infer_opencv.py --model outputs/saved_model
"""
import tensorflow as tf
import cv2
import numpy as np
import argparse
from utils.visualize import decode_predictions, draw_boxes_on_image

def preprocess_frame(frame, image_size=416):
    img = cv2.resize(frame, (image_size,image_size))
    arr = img.astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def run(model_dir):
    model = tf.saved_model.load(model_dir)
    infer = model.signatures.get('serving_default') if hasattr(model, 'signatures') else None
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess_frame(frame)
        if infer is not None:
            out = infer(tf.constant(inp))
            # get first output
            pred = list(out.values())[0].numpy()[0]
        else:
            pred = model.predict(inp)[0]
        boxes = decode_predictions(pred)
        vis = draw_boxes_on_image(inp[0], boxes)
        cv2.imshow('pred', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    args = p.parse_args()
    run(args.model)
