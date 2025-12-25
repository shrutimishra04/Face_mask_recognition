"""tf.data dataset for a YOLO-lite grid detector.
Encodes boxes into grid cells (one box per cell assumption).
"""
import tensorflow as tf
import numpy as np
import json
from PIL import Image

CLASS_NUM = 3
IMAGE_SIZE = 416
GRID_SIZE = 13

def load_records(json_path):
    with open(json_path,'r') as f:
        return json.load(f)

def encode_boxes(boxes, img_w, img_h):
    # boxes: [[xmin,ymin,xmax,ymax,class], ...]
    target = np.zeros((GRID_SIZE, GRID_SIZE, 5 + CLASS_NUM), dtype=np.float32)
    cell_w = img_w / GRID_SIZE
    cell_h = img_h / GRID_SIZE
    for b in boxes:
        xmin, ymin, xmax, ymax, cls = b
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin
        gx = int(cx // cell_w)
        gy = int(cy // cell_h)
        if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
            continue
        # normalized within cell
        tx = (cx - gx*cell_w) / cell_w
        ty = (cy - gy*cell_h) / cell_h
        tw = bw / img_w
        th = bh / img_h
        target[gy, gx, 0:4] = [tx, ty, tw, th]
        target[gy, gx, 4] = 1.0
        target[gy, gx, 5 + cls] = 1.0
    return target

def preprocess_image(image_path, boxes, image_size=IMAGE_SIZE):
    img = Image.open(image_path).convert('RGB')
    w,h = img.size
    img = img.resize((image_size,image_size))
    arr = np.array(img) / 255.0
    target = encode_boxes(boxes, w, h)
    return arr.astype(np.float32), target

def tf_dataset_from_json(json_path, batch=8, shuffle=True):
    records = load_records(json_path)
    def gen():
        for r in records:
            im, tgt = preprocess_image(r['image_path'], r['boxes'])
            yield im, tgt
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32), output_shapes=((IMAGE_SIZE,IMAGE_SIZE,3),(GRID_SIZE,GRID_SIZE,5+CLASS_NUM)))
    if shuffle:
        ds = ds.shuffle(256)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    ds = tf_dataset_from_json(args.json, batch=args.batch)
    for im,t in ds.take(1):
        print(im.shape, t.shape)
