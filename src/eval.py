"""Evaluate model on dataset: compute IoU=0.5 AP per class (simple heuristic).
This is a lightweight evaluator intended for quick checks.
"""
import numpy as np
import tensorflow as tf
from data.dataset import tf_dataset_from_json
from utils.visualize_clean import decode_predictions, visualize_random
import os

def iou(boxA, boxB):
    # boxes: [xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = boxAArea + boxBArea - inter
    return inter / union if union>0 else 0

def evaluate(model, dataset, iou_thresh=0.5):
    # very simple per-class precision/recall
    cls_tp = np.zeros(3)
    cls_fp = np.zeros(3)
    cls_fn = np.zeros(3)
    for im, tgt in dataset:
        preds = model(im).numpy()
        for i in range(im.shape[0]):
            pred = preds[i]
            boxes_p = decode_predictions(pred, conf_thresh=0.25)
            # construct gt boxes from tgt
            gt = tgt[i].numpy()
            gt_boxes = []
            G = gt.shape[0]
            cell_w = 416 / G
            cell_h = 416 / G
            for gy in range(G):
                for gx in range(G):
                    if gt[gy,gx,4] > 0.5:
                        tx,ty,tw,th = gt[gy,gx,0:4]
                        cx = (gx + tx) * cell_w
                        cy = (gy + ty) * cell_h
                        bw = tw * 416
                        bh = th * 416
                        xmin = cx - bw/2; ymin = cy - bh/2; xmax = cx + bw/2; ymax = cy + bh/2
                        cls = int(np.argmax(gt[gy,gx,5:]))
                        gt_boxes.append((xmin,ymin,xmax,ymax,cls))
            matched = [False]*len(gt_boxes)
            for pb in boxes_p:
                px1,py1,px2,py2,conf,cls = pb
                best_i = -1; best_iou = 0
                for j,gb in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    if gb[4] != cls:
                        continue
                    val = iou([px1,py1,px2,py2], gb[:4])
                    if val > best_iou:
                        best_iou = val; best_i = j
                if best_i>=0 and best_iou>=iou_thresh:
                    cls_tp[int(cls)] += 1
                    matched[best_i]=True
                else:
                    cls_fp[int(cls)] += 1
            for j,gb in enumerate(gt_boxes):
                if not matched[j]:
                    cls_fn[gb[4]] += 1
    precision = cls_tp / (cls_tp + cls_fp + 1e-8)
    recall = cls_tp / (cls_tp + cls_fn + 1e-8)
    ap = precision * recall  # rough proxy
    return {'precision':precision.tolist(), 'recall':recall.tolist(), 'ap':ap.tolist()}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--json', required=True)
    p.add_argument('--outdir', default='preds')
    args = p.parse_args()
    model_path = args.model
    # load model: support .h5/.keras via keras, otherwise use tf.saved_model
    model = None
    infer = None
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        loaded = tf.saved_model.load(model_path)
        # use serving_default signature if available
        if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
            infer = loaded.signatures['serving_default']
        else:
            # try to use the callable object
            infer = loaded
    ds = tf_dataset_from_json(args.json, batch=4, shuffle=False)
    # wrap evaluate to call infer/model uniformly
    def model_call(inp):
        if model is not None:
            return model(inp, training=False)
        else:
            out = infer(tf.constant(inp))
            # get first tensor from dict or returned value
            if isinstance(out, dict):
                return list(out.values())[0]
            return out

    stats = {'precision':[], 'recall':[], 'ap':[]}
    # run evaluation
    stats = evaluate(model_call, ds)
    print(stats)
    # save visualization of 10 random samples
    # pass a callable that returns numpy outputs when using SavedModel
    if model is not None:
        vis_model = model
    else:
        vis_model = lambda x: model_call(x).numpy()
    visualize_random(ds, vis_model, n=10, outdir=args.outdir)
    print(f'Wrote predictions to {args.outdir}')
