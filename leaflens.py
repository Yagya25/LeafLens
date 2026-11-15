"""LeafLens: controlled-background leaf spot detection, not disease diagnosis."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def analyze(image):
    if image is None or image.size == 0:
        raise ValueError('Image could not be read.')
    scale = min(1.0, 800 / max(image.shape[:2]))
    work = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    smooth = cv2.GaussianBlur(work, (3, 3), 0)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.inRange(hsv, (10, 45, 30), (100, 255, 255))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k5)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf = np.zeros(work.shape[:2], np.uint8)
    spots = np.zeros_like(leaf)
    if contours:
        cv2.drawContours(leaf, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    leaf_pixels = int(cv2.countNonZero(leaf))
    if leaf_pixels < 0.05 * leaf.size:
        return {'status': 'No usable leaf region', 'spot_count': None,
                'leaf_pixels': leaf_pixels, 'spot_pixels': None,
                'affected_percent': None, 'boxes': [], 'warnings': []}, (work, leaf, spots, work.copy())
    interior = cv2.erode(leaf, k3, iterations=1)
    raw = cv2.inRange(hsv, (0, 65, 25), (32, 255, 190))
    raw = cv2.bitwise_and(raw, interior)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k3)
    raw = cv2.bitwise_and(raw, interior)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    boxes = []
    for label in range(1, n):
        x, y, w, h, area = map(int, stats[label])
        if area >= 20:
            spots[labels == label] = 255
            boxes.append({'x': x, 'y': y, 'width': w, 'height': h, 'pixels': area})
    spot_pixels = int(cv2.countNonZero(spots))
    percent = 100.0 * spot_pixels / leaf_pixels
    annotated = work.copy()
    marked = spots > 0
    annotated[marked] = (0.4 * work[marked] + 0.6 * np.array([40, 40, 240])).astype(np.uint8)
    cv2.drawContours(annotated, [max(contours, key=cv2.contourArea)], -1, (35, 140, 35), 2)
    for i, box in enumerate(boxes, 1):
        x, y, w, h = (box[k] for k in ('x', 'y', 'width', 'height'))
        cv2.rectangle(annotated, (x, y), (x+w-1, y+h-1), (20, 30, 230), 2)
        cv2.putText(annotated, str(i), (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 30, 230), 1, cv2.LINE_AA)
    warnings = []
    if np.any(leaf[0]) or np.any(leaf[-1]) or np.any(leaf[:, 0]) or np.any(leaf[:, -1]):
        warnings.append('Leaf touches image border; recapture the complete leaf.')
    result = {'status': 'Candidate spots detected' if boxes else 'No candidate spots detected',
              'spot_count': len(boxes), 'leaf_pixels': leaf_pixels,
              'spot_pixels': spot_pixels, 'affected_percent': round(percent, 3),
              'boxes': boxes, 'warnings': warnings,
              'working_width': work.shape[1], 'working_height': work.shape[0]}
    return result, (work, leaf, spots, annotated)


def run(input_path, output_dir):
    result, images = analyze(cv2.imread(str(input_path)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in zip(('input', 'leaf_mask', 'spot_mask', 'annotated'), images):
        cv2.imwrite(str(out / (name + '.png')), img)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='One leaf on a plain white or grey background')
    parser.add_argument('--out', default='results/my_leaf')
    args = parser.parse_args()
    run(args.image, args.out)
