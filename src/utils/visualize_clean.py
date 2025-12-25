import numpy as np
import os
import imageio

CLASS_NAMES = ['with_mask', 'without_mask', 'mask_weared_incorrect']


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_predictions(pred, image_size=416, conf_thresh=0.3):
    pred = np.array(pred)
    G = pred.shape[0]
    boxes = []
    for y in range(G):
        for x in range(G):
            cell = pred[y, x]
            conf = float(_sigmoid(cell[4]))
            if conf < conf_thresh:
                continue
            tx, ty, tw, th = [float(v) for v in cell[0:4]]
            cx = (x + tx) * (image_size / G)
            cy = (y + ty) * (image_size / G)
            bw = tw * image_size
            bh = th * image_size
            xmin = max(0, cx - bw / 2)
            ymin = max(0, cy - bh / 2)
            xmax = min(image_size, cx + bw / 2)
            ymax = min(image_size, cy + bh / 2)
            cls = int(np.argmax(cell[5:]))
            boxes.append([xmin, ymin, xmax, ymax, conf, cls])
    return boxes


def draw_boxes_on_image(image, boxes):
    img = (image * 255).astype('uint8').copy()
    import cv2
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return img


def visualize_random(dataset, model, n=10, outdir='preds'):
    os.makedirs(outdir, exist_ok=True)
    ds_iter = iter(dataset)
    for i in range(n):
        im, tgt = next(ds_iter)
        try:
            out = model(im)
        except TypeError:
            try:
                out = model(im, training=False)
            except Exception:
                out = model.predict(im)
        if isinstance(out, dict):
            out = list(out.values())[0]
        out_np = out.numpy() if hasattr(out, 'numpy') else np.array(out)
        pred = out_np[0]
        img_np = im[0].numpy() if hasattr(im[0], 'numpy') else np.array(im[0])
        boxes = decode_predictions(pred)
        vis = draw_boxes_on_image(img_np, boxes)
        imageio.imwrite(os.path.join(outdir, f'pred_{i}.jpg'), vis)
