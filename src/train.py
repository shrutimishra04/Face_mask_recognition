"""Training script using model.fit for the YOLO-lite detector.
Usage example:
  python src/train.py --train_json data/train.json --val_json data/val.json
"""
import tensorflow as tf
from model.yolo_lite import build_yolo_lite, detection_loss
from data.dataset import tf_dataset_from_json, load_records
import argparse
import os

def make_callbacks(outdir):
    cb = []
    cb.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(outdir,'best.h5'), save_best_only=True, save_weights_only=False))
    cb.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3))
    return cb

def main(args):
    model = build_yolo_lite()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=detection_loss)
    train_ds = tf_dataset_from_json(args.train_json, batch=args.batch)
    val_ds = tf_dataset_from_json(args.val_json, batch=args.batch, shuffle=False)
    # compute steps to avoid dataset exhaustion
    train_count = len(load_records(args.train_json))
    val_count = len(load_records(args.val_json))
    steps_per_epoch = max(1, train_count // args.batch)
    validation_steps = max(1, val_count // args.batch)
    os.makedirs(args.out, exist_ok=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=make_callbacks(args.out), steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    # export as SavedModel directory
    tf.saved_model.save(model, os.path.join(args.out, 'saved_model'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_json', required=True)
    p.add_argument('--val_json', required=True)
    p.add_argument('--out', default='outputs')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    main(args)
