from __future__ import annotations

import numpy as np


def iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def average_precision(recall, precision):
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _class_ap(preds, gts, class_id, iou_threshold):
    npos = 0
    gt_cache, used = {}, {}
    for image_id, gt in gts.items():
        boxes = gt["boxes"][gt["labels"] == class_id]
        gt_cache[image_id] = boxes
        used[image_id] = np.zeros(len(boxes), dtype=bool)
        npos += len(boxes)
    if npos == 0:
        return None
    detections = []
    for image_id, pred in preds.items():
        mask = pred["labels"] == class_id
        detections.extend((image_id, box, score) for box, score in zip(pred["boxes"][mask], pred["scores"][mask]))
    detections.sort(key=lambda x: -x[2])
    tp = np.zeros(len(detections)); fp = np.zeros(len(detections))
    for i, (image_id, box, _) in enumerate(detections):
        gt_boxes = gt_cache.get(image_id, np.empty((0, 4)))
        if len(gt_boxes) == 0:
            fp[i] = 1; continue
        overlaps = iou_xyxy(box, gt_boxes)
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not used[image_id][best]:
            tp[i] = 1; used[image_id][best] = True
        else:
            fp[i] = 1
    if not len(detections):
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    return average_precision(tp / npos, tp / np.maximum(tp + fp, 1e-9))


def detection_metrics(preds, gts, class_ids, score_threshold=0.5):
    thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)
    ap50, ap_all = [], []
    for class_id in class_ids:
        value = _class_ap(preds, gts, class_id, 0.50)
        if value is not None: ap50.append(value)
        values = [_class_ap(preds, gts, class_id, t) for t in thresholds]
        values = [v for v in values if v is not None]
        if values: ap_all.append(float(np.mean(values)))

    tp = fp = fn = 0
    for image_id, gt in gts.items():
        pred = preds.get(image_id, {"boxes": np.empty((0, 4)), "labels": np.empty((0,)), "scores": np.empty((0,))})
        keep = pred["scores"] >= score_threshold
        matched = np.zeros(len(gt["boxes"]), dtype=bool)
        for idx in np.argsort(-pred["scores"][keep]):
            boxes = pred["boxes"][keep]; labels = pred["labels"][keep]
            candidates = np.where(gt["labels"] == labels[idx])[0]
            if len(candidates) == 0:
                fp += 1; continue
            overlaps = iou_xyxy(boxes[idx], gt["boxes"][candidates])
            best = int(np.argmax(overlaps)); target = candidates[best]
            if overlaps[best] >= 0.5 and not matched[target]:
                tp += 1; matched[target] = True
            else:
                fp += 1
        fn += int((~matched).sum())
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "mAP50": float(np.mean(ap50)) if ap50 else 0.0, "mAP50_95": float(np.mean(ap_all)) if ap_all else 0.0, "tp": tp, "fp": fp, "fn": fn}
