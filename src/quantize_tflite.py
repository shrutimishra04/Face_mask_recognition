import tensorflow as tf
import argparse

def quantize(saved_model_dir, out_tflite):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_tflite, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--saved', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    quantize(args.saved, args.out)
