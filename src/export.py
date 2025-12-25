import tensorflow as tf
import argparse

def export(saved_h5_or_dir, out_dir):
    model = tf.keras.models.load_model(saved_h5_or_dir, compile=False)
    tf.saved_model.save(model, out_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    export(args.src, args.out)
