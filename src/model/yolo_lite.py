"""YOLO-lite style detection model using a pre-trained MobileNetV2 backbone.
Output shape: (GRID_SIZE, GRID_SIZE, 5 + CLASS_NUM)
"""
import tensorflow as tf
from tensorflow.keras import layers

GRID_SIZE = 13
CLASS_NUM = 3

def build_yolo_lite(input_shape=(416,416,3)):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base.output
    # reduce channels and predict per-grid output
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    # ensure spatial dims are GRID_SIZE x GRID_SIZE
    x = layers.Conv2D(5 + CLASS_NUM, 1, padding='same')(x)
    # if necessary, resize to GRID_SIZE
    def resize_to_grid(t):
        return tf.image.resize(t, (GRID_SIZE, GRID_SIZE))
    x = layers.Lambda(resize_to_grid)(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

def detection_loss(y_true, y_pred):
    # y_true/y_pred shape: (B, G, G, 5+C)
    # box coords MSE for cells with object
    obj_mask = y_true[...,4:5]
    # box coords
    box_true = y_true[...,0:4]
    box_pred = y_pred[...,0:4]
    b_loss = tf.reduce_sum(obj_mask * tf.square(box_true - box_pred))
    # confidence loss
    conf_true = y_true[...,4]
    conf_pred = tf.sigmoid(y_pred[...,4])
    c_loss = tf.reduce_sum(tf.square(conf_true - conf_pred))
    # class loss - categorical crossentropy per cell
    cls_true = y_true[...,5:]
    cls_pred = tf.nn.softmax(y_pred[...,5:])
    cls_loss = tf.reduce_sum(obj_mask * tf.reduce_sum(-cls_true * tf.math.log(1e-8 + cls_pred), axis=-1, keepdims=True))
    return b_loss + c_loss + cls_loss
