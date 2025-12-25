import json
import random
import argparse
import os

def split_json(in_json, out_dir, ratios=(0.7,0.15,0.15), seed=42):
    with open(in_json,'r') as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    n1 = int(ratios[0]*n)
    n2 = n1 + int(ratios[1]*n)
    splits = {'train': data[:n1], 'val': data[n1:n2], 'test': data[n2:]}
    os.makedirs(out_dir, exist_ok=True)
    for k,v in splits.items():
        with open(os.path.join(out_dir, f'{k}.json'),'w') as f:
            json.dump(v,f)
    return splits

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_json', required=True)
    p.add_argument('--outdir', required=True)
    args = p.parse_args()
    split_json(args.in_json, args.outdir)
