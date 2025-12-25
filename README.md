# Face Mask Detection (YOLO-lite style)

This repo provides a minimal end-to-end pipeline for training a YOLO-lite style CNN detector for face-mask detection.

Features:
- Parse Pascal VOC XML annotations
- tf.data pipeline with augmentations
- MobileNetV2 backbone + detection head (grid-based)
- Training with box MSE + categorical crossentropy
- Evaluation at IoU=0.5 per class
- Visualization utilities
- Export as TensorFlow SavedModel and OpenCV inference script

Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

See `src/` for scripts. Run `pip install -r requirements.txt`.