"""Parse Pascal VOC XML annotations into a simple JSON list.
Produces a list of records: {'image_path':..., 'boxes': [[xmin,ymin,xmax,ymax, class_id], ...]}
"""
import os
import json
from lxml import etree

CLASS_MAP = {"with_mask":0, "without_mask":1, "mask_weared_incorrect":2}

def parse_xml(xml_path, images_dir=None):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    fname = root.findtext('filename')
    if images_dir:
        image_path = os.path.join(images_dir, fname)
    else:
        image_path = fname
    boxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        cls = CLASS_MAP.get(name, None)
        if cls is None:
            continue
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.findtext('xmin')))
        ymin = int(float(bnd.findtext('ymin')))
        xmax = int(float(bnd.findtext('xmax')))
        ymax = int(float(bnd.findtext('ymax')))
        boxes.append([xmin, ymin, xmax, ymax, cls])
    return {'image_path': image_path, 'boxes': boxes}

def build_annotation_list(annotations_dir, images_dir=None, out_json=None):
    records = []
    for fname in os.listdir(annotations_dir):
        if not fname.endswith('.xml'):
            continue
        xmlp = os.path.join(annotations_dir, fname)
        rec = parse_xml(xmlp, images_dir)
        if rec['boxes']:
            records.append(rec)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(records, f)
    return records

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ann', required=True)
    p.add_argument('--imgdir', required=False)
    p.add_argument('--out', required=False)
    args = p.parse_args()
    recs = build_annotation_list(args.ann, args.imgdir, args.out)
    print(f'Parsed {len(recs)} records')
