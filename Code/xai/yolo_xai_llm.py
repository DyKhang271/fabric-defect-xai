import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from xai_metrics import (
    comparison_report,
    deletion_auc,
    faithfulness_test,
    gradcam_heatmap,
    gradcam_multi,
    insertion_auc,
    pointing_game,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
XAI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = XAI_DIR.parents[1]


@dataclass
class DetBox:
    cls: int
    name: str
    conf: float
    xyxy: list


def load_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def load_defect_knowledge(path):
    if not path:
        return {}
    knowledge_path = Path(path)
    if not knowledge_path.exists():
        print(f"Knowledge file not found, continuing without it: {knowledge_path}")
        return {}
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    defects = data.get("defects", data)
    return {str(k): v for k, v in defects.items()}


def knowledge_for_class(knowledge, cls_id, name):
    return knowledge.get(str(cls_id)) or knowledge.get(str(name)) or {}


def predict_boxes(model, image, names, imgsz, conf, iou, device):
    kwargs = {"imgsz": imgsz, "conf": conf, "iou": iou, "verbose": False}
    if device is not None:
        kwargs["device"] = device

    result = model.predict(image, **kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = []
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        boxes.append(
            DetBox(
                cls=cls_id,
                name=names.get(cls_id, str(cls_id)),
                conf=float(result.boxes.conf[i].item()),
                xyxy=[float(x) for x in result.boxes.xyxy[i].detach().cpu().numpy()],
            )
        )
    return sorted(boxes, key=lambda b: b.conf, reverse=True)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def image_to_label_path(image_path):
    image_path = Path(image_path)
    text = str(image_path)
    marker = f"{os.sep}images{os.sep}"
    if marker in text:
        return Path(text.replace(marker, f"{os.sep}labels{os.sep}")).with_suffix(".txt")

    stems = [image_path.stem]
    stem = image_path.stem
    if (
        len(stem) > 16
        and stem[8] == "_"
        and stem[15] == "_"
        and stem[:8].isdigit()
        and stem[9:15].isdigit()
    ):
        stems.append(stem[16:])

    labels_root = PROJECT_ROOT / "voc2007_yolo" / "labels"
    for candidate_stem in dict.fromkeys(stems):
        for split in ("test", "val", "train"):
            candidate = labels_root / split / f"{candidate_stem}.txt"
            if candidate.exists():
                return candidate
    return None


def read_yolo_labels(label_path, w, h, names):
    if not label_path or not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes.append(DetBox(cls_id, names.get(cls_id, str(cls_id)), 1.0, [x1, y1, x2, y2]))
    return boxes


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return [
        int(max(0, min(w - 1, round(x1)))),
        int(max(0, min(h - 1, round(y1)))),
        int(max(0, min(w - 1, round(x2)))),
        int(max(0, min(h - 1, round(y2)))),
    ]


def draw_boxes(image, boxes, color, prefix):
    out = image.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = clamp_box(box.xyxy, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix} {box.name}"
        if box.conf < 1.0:
            label += f" {box.conf:.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def target_score(model, image, names, target, args):
    boxes = predict_boxes(model, image, names, args.imgsz, args.conf, args.iou, args.device)
    match_iou = float(getattr(args, "match_iou", 0.30))
    scores = [
        box.conf
        for box in boxes
        if box.cls == target.cls and bbox_iou(box.xyxy, target.xyxy) >= match_iou
    ]
    return max(scores) if scores else 0.0


def scan_positions(length, start, end, patch, stride):
    start = int(max(0, min(length, round(start))))
    end = int(max(start + 1, min(length, round(end))))
    patch = max(1, int(patch))
    stride = max(1, int(stride))
    if patch >= length:
        return [0]
    if patch >= end - start:
        center = (start + end) // 2
        return [max(0, min(length - patch, center - patch // 2))]

    last = max(start, end - patch)
    positions = list(range(start, last + 1, stride))
    if not positions or positions[-1] != last:
        positions.append(last)
    return sorted(set(max(0, min(length - patch, p)) for p in positions))

def scan_grid_for_target(image, target, args):
    h, w = image.shape[:2]
    if getattr(args, "xai_scope", "full") != "bbox":
        return (
            scan_positions(h, 0, h, args.patch, args.stride),
            scan_positions(w, 0, w, args.patch, args.stride),
        )

    x1, y1, x2, y2 = target.xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scale = max(1.0, float(getattr(args, "roi_scale", 3.0)))
    min_roi = max(float(getattr(args, "min_roi", args.patch)), float(args.patch))
    roi_w = max(bw * scale, min_roi)
    roi_h = max(bh * scale, min_roi)
    return (
        scan_positions(h, cy - roi_h / 2.0, cy + roi_h / 2.0, args.patch, args.stride),
        scan_positions(w, cx - roi_w / 2.0, cx + roi_w / 2.0, args.patch, args.stride),
    )

def occlusion_heatmap(model, image, names, target, args):
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    if getattr(args, "fill_mode", "gray") == "mean":
        fill = image.mean(axis=(0, 1)).astype(np.uint8)
    else:
        fill = np.array([127, 127, 127], dtype=np.uint8)

    base_score = max(target.conf, 1e-8)
    y_positions, x_positions = scan_grid_for_target(image, target, args)
    total = len(y_positions) * len(x_positions)
    done = 0

    for y in y_positions:
        for x in x_positions:
            y2 = min(h, y + args.patch)
            x2 = min(w, x + args.patch)
            occluded = image.copy()
            occluded[y:y2, x:x2] = fill
            new_score = target_score(model, occluded, names, target, args)
            drop = max(0.0, base_score - new_score)
            heat[y:y2, x:x2] += drop
            counts[y:y2, x:x2] += 1.0
            done += 1
            if args.progress and (done == 1 or done % 25 == 0 or done == total):
                print(f"  patches {done}/{total}", flush=True)

    heat = heat / np.maximum(counts, 1.0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def heat_ratio(heat, box):
    if heat is None:
        return None
    h, w = heat.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1 or heat.sum() <= 0:
        return 0.0
    return float(heat[y1:y2, x1:x2].sum() / heat.sum())


def xai_method_full_name(method):
    if method == "occlusion":
        return "Occlusion Sensitivity Analysis"
    if method == "gradcam":
        return "Grad-CAM"
    return "Occlusion Sensitivity Analysis + Grad-CAM"


def heat_ratio_definition():
    return "sum(heatmap inside target/GT box) / sum(heatmap over full image)"


def xai_parameters(args):
    return {
        "patch_size": int(getattr(args, "patch", 0)),
        "stride": int(getattr(args, "stride", 0)),
        "fill_mode": getattr(args, "fill_mode", "gray"),
        "xai_scope": getattr(args, "xai_scope", "full"),
        "roi_scale": float(getattr(args, "roi_scale", 3.0)),
        "min_roi": int(getattr(args, "min_roi", 0)),
        "confidence_threshold": float(getattr(args, "conf", 0.0)),
        "nms_iou_threshold": float(getattr(args, "iou", 0.0)),
    }


def target_score_metadata(target, args, skipped=False):
    return {
        "original_score": None if target is None or skipped else float(target.conf),
        "score_definition": (
            "YOLO confidence of the same-class box matched to the original target by IoU"
        ),
        "importance_formula": "Importance_i = max(0, S0 - Si); heatmap normalized to [0, 1]",
        "missing_box_score": 0.0,
        "skipped": bool(skipped),
    }


def matching_rule(args):
    return {
        "same_class": True,
        "iou_threshold": float(getattr(args, "match_iou", 0.30)),
        "missing_match_score": 0.0,
    }


def llm_role_note():
    return (
        "Optional qualitative interpretation only; quantitative evidence comes from "
        "YOLO metrics, IoU, TP/FP/FN/TN, and XAI metrics."
    )


def add_scientific_metadata(analysis, target, args, skipped=False):
    method = getattr(args, "xai_method", "both")
    analysis.update({
        "xai_method_full_name": xai_method_full_name(method),
        "xai_parameters": xai_parameters(args),
        "target_definition": "one YOLO predicted bounding box: class, confidence, xyxy bbox",
        "target_score": target_score_metadata(target, args, skipped=skipped),
        "matching_rule": matching_rule(args),
        "heat_ratio_definition": heat_ratio_definition(),
        "llm_role": llm_role_note(),
    })
    return analysis


def xai_method_enabled(args, method):
    selected = getattr(args, "xai_method", "both")
    return selected == "both" or selected == method


def combine_heatmaps(heatmaps, h, w):
    valid = [heat for heat in heatmaps if heat is not None]
    if not valid:
        return None
    combined = np.maximum.reduce(valid)
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined.astype(np.float32)


def reliability_label(score):
    if score is None:
        return "Không có dữ liệu"
    if score >= 0.45:
        return "Cao"
    if score >= 0.25:
        return "Trung bình"
    return "Thấp"


def xai_reliability_score(pred_ratio, gt_ratio=None, pointing=None):
    if gt_ratio is not None:
        point_score = float(pointing) if pointing is not None else 0.0
        return max(0.0, min(1.0, 0.7 * float(gt_ratio) + 0.3 * point_score))
    if pred_ratio is None:
        return None
    return max(0.0, min(1.0, float(pred_ratio)))


def xai_method_summary(method, runtime_sec, pred_ratio, gt_ratio=None, pointing=None):
    score = xai_reliability_score(pred_ratio, gt_ratio, pointing)
    return {
        "method": method,
        "runtime_sec": round(float(runtime_sec or 0.0), 3),
        "pred_bbox_ratio": None if pred_ratio is None else round(float(pred_ratio), 4),
        "gt_bbox_ratio": None if gt_ratio is None else round(float(gt_ratio), 4),
        "pointing_game": None if pointing is None else float(pointing),
        "reliability_score": None if score is None else round(float(score), 4),
        "reliability_label": reliability_label(score),
    }


def rank_xai_rows(rows):
    if len(rows) > 1:
        fastest = min(rows, key=lambda row: row["runtime_sec"])
        strongest = max(
            rows,
            key=lambda row: -1.0 if row["reliability_score"] is None else row["reliability_score"],
        )
        for row in rows:
            row["speed_rank"] = "Nhanh nhất" if row is fastest else ""
            row["reliability_rank"] = "Đáng tin cậy nhất" if row is strongest else ""
    return rows


def xai_comparison_from_heats(heat_occ, heat_gc, target, gt, w, h, runtimes):
    rows = []
    if heat_occ is not None:
        occ_gt = heat_ratio(heat_occ, gt.xyxy) if gt else None
        occ_point = pointing_game(heat_occ, gt.xyxy, w, h) if gt else None
        rows.append(xai_method_summary(
            "Occlusion", runtimes.get("occlusion", 0.0),
            heat_ratio(heat_occ, target.xyxy), occ_gt, occ_point
        ))
    if heat_gc is not None:
        gc_gt = heat_ratio(heat_gc, gt.xyxy) if gt else None
        gc_point = pointing_game(heat_gc, gt.xyxy, w, h) if gt else None
        rows.append(xai_method_summary(
            "Grad-CAM", runtimes.get("gradcam", 0.0),
            heat_ratio(heat_gc, target.xyxy), gc_gt, gc_point
        ))
    return rank_xai_rows(rows)


def save_heatmap_outputs(image, heat, out_dir, stem, preds, gts, target):
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.58, heat_color, 0.42, 0)
    overlay = draw_boxes(overlay, preds, (0, 255, 0), "loi")
    overlay = draw_boxes(overlay, [target], (0, 255, 255), "vung chinh")
    pred_img = draw_boxes(image, preds, (0, 255, 0), "loi")
    gt_img = draw_boxes(image, gts, (255, 0, 0), "gt")

    paths = {
        "original": out_dir / f"{stem}_original.jpg",
        "pred": out_dir / f"{stem}_pred.jpg",
        "gt": out_dir / f"{stem}_gt.jpg",
        "heatmap": out_dir / f"{stem}_heatmap.jpg",
        "overlay": out_dir / f"{stem}_xai_overlay.jpg",
    }
    cv2.imwrite(str(paths["original"]), image)
    cv2.imwrite(str(paths["pred"]), pred_img)
    cv2.imwrite(str(paths["gt"]), gt_img)
    cv2.imwrite(str(paths["heatmap"]), heat_color)
    cv2.imwrite(str(paths["overlay"]), overlay)
    return paths


def save_multi_heatmap_outputs(image, heat, out_dir, stem, preds, gts, targets):
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.58, heat_color, 0.42, 0)
    overlay = draw_boxes(overlay, preds, (0, 255, 0), "loi")
    overlay = draw_boxes(overlay, targets, (0, 255, 255), "giai thich")
    pred_img = draw_boxes(image, preds, (0, 255, 0), "loi")
    gt_img = draw_boxes(image, gts, (255, 0, 0), "gt")

    paths = {
        "original": out_dir / f"{stem}_original.jpg",
        "pred": out_dir / f"{stem}_pred.jpg",
        "gt": out_dir / f"{stem}_gt.jpg",
        "heatmap": out_dir / f"{stem}_heatmap.jpg",
        "overlay": out_dir / f"{stem}_xai_overlay.jpg",
    }
    cv2.imwrite(str(paths["original"]), image)
    cv2.imwrite(str(paths["pred"]), pred_img)
    cv2.imwrite(str(paths["gt"]), gt_img)
    cv2.imwrite(str(paths["heatmap"]), heat_color)
    cv2.imwrite(str(paths["overlay"]), overlay)
    return paths


def save_no_prediction_outputs(image, out_dir, stem, gts):
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    pred_img = image.copy()
    gt_img = draw_boxes(image, gts, (255, 0, 0), "gt")
    overlay = gt_img.copy()

    paths = {
        "original": out_dir / f"{stem}_original.jpg",
        "pred": out_dir / f"{stem}_pred.jpg",
        "gt": out_dir / f"{stem}_gt.jpg",
        "heatmap": out_dir / f"{stem}_heatmap.jpg",
        "overlay": out_dir / f"{stem}_xai_overlay.jpg",
    }
    cv2.imwrite(str(paths["original"]), image)
    cv2.imwrite(str(paths["pred"]), pred_img)
    cv2.imwrite(str(paths["gt"]), gt_img)
    cv2.imwrite(str(paths["heatmap"]), heat_color)
    cv2.imwrite(str(paths["overlay"]), overlay)
    return paths


def save_gradcam_outputs(image, heat_gc, out_dir, stem, preds, targets):
    gc_u8 = np.uint8(255 * heat_gc)
    gc_color = cv2.applyColorMap(gc_u8, cv2.COLORMAP_JET)
    gc_overlay = cv2.addWeighted(image, 0.58, gc_color, 0.42, 0)
    gc_overlay = draw_boxes(gc_overlay, preds, (0, 255, 0), "loi")
    gc_overlay = draw_boxes(gc_overlay, targets, (0, 255, 255), "vung chinh")
    gc_paths = {
        "gradcam": out_dir / f"{stem}_gradcam.jpg",
        "gradcam_overlay": out_dir / f"{stem}_gradcam_overlay.jpg",
    }
    cv2.imwrite(str(gc_paths["gradcam"]), gc_color)
    cv2.imwrite(str(gc_paths["gradcam_overlay"]), gc_overlay)
    return gc_paths


def data_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def clean_customer_text(value):
    if not value:
        return ""
    text = " ".join(str(value).replace("\n", " ").split())
    for tag in ("[PPT]", "[YOLO]", "[CG]"):
        text = text.replace(tag, "")
    text = text.replace("theo chiều dọc vải (hướng warp)", "theo chiều dọc vải")
    text = text.replace("(hướng warp)", "(chiều dọc)")
    text = text.replace("(warp direction)", "(chiều dọc)")
    text = text.replace("hướng warp", "chiều dọc")
    text = text.replace("weft", "chiều ngang")
    banned = (
        "bbox", "yolo", "xai", "heatmap", "median", "ratio", "p10", "p90",
        "dataset", "annotation", "slide", "ppt", "bl mode", "df mode",
        "confidence", "w=", "h=", "w~", "h~", "class", "mô hình",
        "~", "%", "chiều cao ảnh", "chiều rộng ảnh",
    )
    parts = []
    for sentence in text.replace(";", ".").split("."):
        s = sentence.strip(" -")
        if not s:
            continue
        if any(token in s.lower() for token in banned):
            continue
        parts.append(s)
    return ". ".join(parts[:2])


def confidence_label(conf):
    if conf >= 0.75:
        return "cao"
    if conf >= 0.50:
        return "trung bình"
    return "thấp, cần kiểm tra lại"


def compact_knowledge_for_llm(args, analysis):
    items = []
    seen = set()
    detections = analysis.get("detections") or [
        {"target": analysis["target"], "defect_knowledge": analysis.get("defect_knowledge") or {}}
    ]
    for det in detections:
        target = det["target"]
        key = target["name"]
        if key in seen:
            continue
        seen.add(key)
        k = det.get("defect_knowledge") or {}
        items.append({
            "ma_loi_noi_bo": key,
            "ten_hien_thi_bat_buoc": k.get("ten_hien_thi") or key,
            "mo_ta_ngan": k.get("mo_ta_ngan", ""),
            "dau_hieu_anh": k.get("dau_hieu_anh", []),
            "tinh_chat": k.get("tinh_chat", ""),
            "muc_do": k.get("muc_do", ""),
            "huong_xu_ly": k.get("huong_xu_ly", ""),
            "de_nham_voi": k.get("de_nham_voi", []),
            "muc_tin_cay": confidence_label(float(target.get("conf", 0.0))),
        })
    return {
        "file_da_doc": str(getattr(args, "knowledge", "")),
        "quy_tac_bat_buoc": [
            "Chỉ dùng tên lỗi trong ten_hien_thi_bat_buoc, không tự đổi sang tên gọi khác.",
            "Mô tả, ý nghĩa và khuyến nghị phải dựa trên mo_ta_ngan, dau_hieu_anh, tinh_chat, muc_do, huong_xu_ly.",
            "Vị trí phải dựa trên vùng màu nổi bật và vùng khoanh trong ảnh kết quả.",
        ],
        "loi_lien_quan": items,
    }


def normalize_llm_answer(text):
    if not text:
        return text
    cleaned = str(text).replace("**", "").replace("__", "")
    cleaned = cleaned.replace("###", "").replace("##", "").replace("#", "")
    lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ", "• ")):
            s = s[2:].strip()
        lines.append(s)
    return "\n".join(lines).strip()


def customer_metadata(analysis):
    target = analysis.get("target") or {"name": "No prediction", "conf": 0.0}
    knowledge = analysis.get("defect_knowledge") or {}
    signs = []
    for item in knowledge.get("dau_hieu_anh") or []:
        raw = str(item).lower()
        if any(token in raw for token in ("[yolo]", "bbox", "median", "ratio", "p10", "p90", "w=", "h=", "~", "%")):
            continue
        cleaned = clean_customer_text(item)
        if cleaned:
            signs.append(cleaned)
    return {
        "ten_loi": knowledge.get("ten_hien_thi") or target["name"],
        "ma_loi_noi_bo": target["name"],
        "muc_tin_cay": confidence_label(float(target["conf"])),
        "mo_ta_ngan": clean_customer_text(knowledge.get("mo_ta_ngan")),
        "dau_hieu_de_hieu": signs[:3],
        "muc_do": clean_customer_text(knowledge.get("muc_do")),
        "khuyen_nghi_khach_hang": (
            "Nên giữ lại ảnh kết quả, đối chiếu vùng được đánh dấu, "
            "sau đó chuyển mẫu sang bộ phận kiểm tra chất lượng để xác nhận trước khi phân loại hoặc xuất hàng."
        ),
    }


def discover_vision_model(base_url, api_key):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(f"{base_url}/models/image-to-text", headers=headers)
        with urllib.request.urlopen(req, timeout=20) as res:
            data = json.loads(res.read().decode("utf-8"))
        models = data.get("data", [])
        return models[0].get("id") if models else None
    except Exception:
        return None


def llm_config(args):
    base_url = args.llm_base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("NINEROUTER_URL"):
        base_url = os.getenv("NINEROUTER_URL").rstrip("/") + "/v1"
    if not base_url and os.getenv("OPENAI_API_KEY"):
        base_url = "https://api.openai.com/v1"
    if not base_url:
        return None, None, None

    base_url = base_url.rstrip("/")
    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("NINEROUTER_KEY") or ""
    model = args.llm_model or os.getenv("XAI_LLM_MODEL") or discover_vision_model(base_url, api_key)
    if not model and "openai.com" in base_url:
        model = "gpt-4o-mini"
    return base_url, api_key, model


def ask_vision_llm(args, paths, analysis):
    base_url, api_key, model = llm_config(args)
    if not base_url or not model:
        return None

    if args.audience == "engineer":
        prompt = (
            "You are an explainable-AI assistant for YOLO defect detection. "
            "Answer in Vietnamese. Use the original image, prediction image, raw heatmap, XAI heatmap overlay, "
            "and Grad-CAM overlay when provided. "
            "Do not invent labels outside the metadata. "
            "Use 'defect_knowledge' as the trusted class glossary. If it is empty or has TODO fields, say that domain knowledge is incomplete. "
            "Explain: detected defect, confidence, where the heatmap focuses, whether it supports the prediction, "
            "and one caution if the heatmap is weak or off-object.\n\n"
            f"Metadata JSON:\n{json.dumps(analysis, ensure_ascii=False, indent=2)}"
        )
        system = "You explain computer-vision model results clearly and cautiously."
    else:
        safe = customer_metadata(analysis)
        knowledge_pack = compact_knowledge_for_llm(args, analysis)
        response_format = (
            "ĐỊNH DẠNG BẮT BUỘC - chỉ trả lời đúng 6 dòng, không Markdown, không in đậm, không bullet:\n"
            "KẾT QUẢ: <dùng đúng tên lỗi trong ten_hien_thi_bat_buoc và mức tin cậy>\n"
            "VỊ TRÍ: <mô tả vị trí dựa trên vùng màu nổi bật/vùng khoanh trong ảnh>\n"
            "DẤU HIỆU: <dấu hiệu quan sát được, đối chiếu dau_hieu_anh trong YAML>\n"
            "Ý NGHĨA: <giải thích ngắn từ mo_ta_ngan/tinh_chat/muc_do trong YAML>\n"
            "KHUYẾN NGHỊ: <dựa trên huong_xu_ly trong YAML, viết cho khách hàng>\n"
            "LƯU Ý: <nếu chưa chắc, yêu cầu nhân viên kiểm tra chất lượng xác nhận>\n"
        )
        prompt = (
            "Bạn là trợ lý kiểm tra chất lượng vải hiển thị trực tiếp cho khách hàng. "
            "Hãy trả lời bằng tiếng Việt tự nhiên, lịch sự, dễ hiểu. "
            "Không dùng thuật ngữ kỹ thuật như YOLO, bbox, heatmap, XAI, confidence score, dataset, slide, BL/DF. "
            "Dựa vào vùng màu nổi bật và vùng khoanh trên ảnh để giải thích. "
            "Có thể gọi vùng màu/vùng khoanh trên ảnh là 'vùng được hệ thống đánh dấu'. "
            "Bạn bắt buộc phải dùng gói tri thức được đọc từ file defect_knowledge.yaml dưới đây làm nguồn đúng. "
            "Không tự đặt tên lỗi mới, không dịch/diễn giải tên lỗi khác với ten_hien_thi_bat_buoc. "
            "Không hướng dẫn kỹ sư IT, không nói về cách sửa mô hình. "
            "Nếu mức tin cậy thấp, nói nhẹ nhàng rằng cần nhân viên kiểm tra lại. "
            "Nêu rõ lỗi nằm ở vùng nào trên ảnh nếu thấy được. "
            f"{response_format}\n"
            f"Gói tri thức đã đọc từ YAML:\n{json.dumps(knowledge_pack, ensure_ascii=False, indent=2)}\n\n"
            f"Thông tin an toàn cho khách hàng:\n{json.dumps(safe, ensure_ascii=False, indent=2)}"
        )
        system = (
            "Bạn viết thông báo kiểm tra chất lượng cho khách hàng. "
            "Tuân thủ đúng định dạng 6 dòng. Không dùng Markdown. "
            "Nguồn kiến thức bắt buộc là dữ liệu YAML do người dùng cung cấp trong prompt."
        )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url(paths["original"])}},
        {"type": "image_url", "image_url": {"url": data_url(paths["pred"])}},
        {"type": "image_url", "image_url": {"url": data_url(paths["heatmap"])}},
        {"type": "image_url", "image_url": {"url": data_url(paths["overlay"])}},
    ]
    if paths.get("gradcam_overlay"):
        content.append({"type": "image_url", "image_url": {"url": data_url(paths["gradcam_overlay"])}})

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": content,
            },
        ],
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.llm_timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
        return normalize_llm_answer(data["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return f"LLM error {e.code}: {detail}"
    except Exception as e:
        return f"LLM error: {e}"


def deterministic_report(analysis, audience="customer"):
    t = analysis.get("target") or {"name": "No prediction", "conf": 0.0}
    det_eval = analysis.get("detection_evaluation") or {}
    if not det_eval.get("has_prediction", True):
        case_type = det_eval.get("case_type", "FN")
        if audience == "customer":
            if case_type == "FN":
                return "\n".join([
                    "Ket qua: He thong chua phat hien loi voi nguong hien tai.",
                    "Doi chieu GT: Anh co annotation loi that, nen day la False Negative.",
                    "Khuyen nghi: Giam nguong phat hien hoac kiem tra lai anh/nhan de tranh bo sot loi.",
                ])
            return "\n".join([
                "Ket qua: He thong khong phat hien loi tren anh.",
                "Doi chieu GT: Anh khong co annotation loi, nen day la True Negative.",
                "Khuyen nghi: Co the luu ket qua lam mau anh dat neu quy trinh kiem tra xac nhan lai.",
            ])
        return "\n".join([
            f"Ket qua detection: {case_type}.",
            f"So prediction: {analysis.get('num_predictions', 0)}.",
            f"So ground truth: {analysis.get('num_ground_truth', 0)}.",
            "XAI skipped: khong co predicted bbox target de giai thich.",
        ])

    if audience == "customer":
        detections = analysis.get("detections") or []
        if len(detections) > 1:
            names = []
            for det in detections:
                item = {
                    "target": det["target"],
                    "defect_knowledge": det.get("defect_knowledge") or {},
                }
                meta_item = customer_metadata(item)
                names.append(f"{meta_item['ten_loi']} ({meta_item['muc_tin_cay']})")
            primary_meta = customer_metadata(analysis)
            return "\n".join(
                [
                    f"Kết quả: Hệ thống phát hiện {len(detections)} vùng bất thường trên ảnh.",
                    "Các dấu hiệu: " + "; ".join(names) + ".",
                    (
                        f"Vùng được làm nổi bật: đang ưu tiên {analysis.get('num_explained', len(detections))} "
                        f"trong {len(detections)} vùng đã phát hiện."
                    ),
                    f"Ý nghĩa chính: {primary_meta['mo_ta_ngan'] or 'Đây là dấu hiệu bất thường trên bề mặt vải cần được kiểm tra lại.'}.",
                    f"Khuyến nghị: {primary_meta['khuyen_nghi_khach_hang']}",
                ]
            )

        meta = customer_metadata(analysis)
        return "\n".join(
            [
                f"Kết quả: Hệ thống phát hiện dấu hiệu {meta['ten_loi']} trên ảnh.",
                f"Mức tin cậy: {meta['muc_tin_cay']}.",
                f"Ý nghĩa: {meta['mo_ta_ngan'] or 'Đây là dấu hiệu bất thường trên bề mặt vải cần được kiểm tra lại.'}.",
                f"Khuyến nghị: {meta['khuyen_nghi_khach_hang']}",
            ]
        )

    pred_ratio = analysis["heat_in_pred_bbox_ratio"]
    gt_ratio = analysis.get("heat_in_gt_bbox_ratio")
    k = analysis.get("defect_knowledge") or {}
    lines = [
        f"Ket qua: model du doan lop {t['name']} voi confidence {t['conf']:.3f}.",
        f"Heatmap tap trung {pred_ratio:.1%} nang luong trong bbox du doan.",
    ]
    if k:
        lines.append(f"Tri thuc lop: {k.get('ten_hien_thi', t['name'])} - {k.get('mo_ta_ngan', 'chua co mo ta')}.")
    if gt_ratio is not None:
        lines.append(f"So voi nhan that, heatmap nam {gt_ratio:.1%} trong bbox ground truth.")
    if pred_ratio >= 0.45:
        lines.append("Nhan xet: vung XAI kha ung voi doi tuong duoc detect, prediction co tinh hop ly.")
    elif pred_ratio >= 0.25:
        lines.append("Nhan xet: vung XAI co lien quan mot phan, nen xem lai anh overlay va bbox.")
    else:
        lines.append("Nhan xet: heatmap yeu hoac lech bbox, can canh giac model co the dua vao nen/nhieu.")
    return "\n".join(lines)


def match_gt(target, gts):
    same_cls = [gt for gt in gts if gt.cls == target.cls]
    if not same_cls:
        return None, 0.0
    best = max(same_cls, key=lambda gt: bbox_iou(target.xyxy, gt.xyxy))
    return best, bbox_iou(target.xyxy, best.xyxy)


def eval_iou_threshold(args):
    return float(getattr(args, "eval_iou", 0.50))


def detection_evaluation_for_target(target, gts, args):
    gt, gt_iou = match_gt(target, gts)
    threshold = eval_iou_threshold(args)
    case_type = "TP" if gt and gt_iou >= threshold else "FP"
    return gt, gt_iou, {
        "case_type": case_type,
        "iou_pred_gt": round(float(gt_iou), 4),
        "iou_threshold": threshold,
        "has_prediction": True,
        "has_ground_truth": bool(gts),
        "matched_gt": asdict(gt) if gt else None,
        "note": "TP when same-class predicted box matches GT with IoU >= threshold; otherwise FP.",
    }


def no_prediction_evaluation(gts, args):
    threshold = eval_iou_threshold(args)
    case_type = "FN" if gts else "TN"
    return {
        "case_type": case_type,
        "iou_pred_gt": 0.0,
        "iou_threshold": threshold,
        "has_prediction": False,
        "has_ground_truth": bool(gts),
        "matched_gt": None,
        "note": "No prediction: FN if GT exists, TN if no GT exists.",
    }


def detection_summary(preds, gts, args):
    threshold = eval_iou_threshold(args)
    if not preds:
        counts = {"TP": 0, "FP": 0, "FN": len(gts), "TN": 0 if gts else 1}
        return {
            "image_case_type": "FN" if gts else "TN",
            "case_counts": counts,
            "iou_threshold": threshold,
            "num_predictions": 0,
            "num_ground_truth": len(gts),
        }

    matched_pred = set()
    matched_gt = set()
    for pi, pred in enumerate(preds):
        candidates = [
            (bbox_iou(pred.xyxy, gt.xyxy), gi)
            for gi, gt in enumerate(gts)
            if gt.cls == pred.cls
        ]
        if not candidates:
            continue
        best_iou, gi = max(candidates, key=lambda item: item[0])
        if best_iou >= threshold and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    if tp and not fp and not fn:
        image_case = "TP"
    elif not tp and fp and not fn:
        image_case = "FP"
    elif not tp and not fp and fn:
        image_case = "FN"
    elif not tp and fp and fn:
        image_case = "FP+FN"
    else:
        image_case = "TP+FP/FN"
    return {
        "image_case_type": image_case,
        "case_counts": {"TP": tp, "FP": fp, "FN": fn, "TN": 0},
        "iou_threshold": threshold,
        "num_predictions": len(preds),
        "num_ground_truth": len(gts),
    }


def compute_xai_metrics(model, image, names, target, args, heat, heat_gc, gt, w, h, prefix="  "):
    xai_metrics = {}
    if gt and heat is not None:
        xai_metrics["pointing_game_occlusion"] = pointing_game(heat, gt.xyxy, w, h)
    if gt and heat_gc is not None:
        xai_metrics["pointing_game_gradcam"] = pointing_game(heat_gc, gt.xyxy, w, h)
    if getattr(args, "eval_metrics", False) and heat is not None:
        print(f"{prefix}Deletion AUC ...", flush=True)
        del_auc, _ = deletion_auc(model, image, names, target, heat, target_score, args)
        print(f"{prefix}Insertion AUC ...", flush=True)
        ins_auc, _ = insertion_auc(model, image, names, target, heat, target_score, args)
        xai_metrics["deletion_auc"] = del_auc
        xai_metrics["insertion_auc"] = ins_auc
        print(f"{prefix}Faithfulness test ...", flush=True)
        faith = faithfulness_test(model, image, names, target, heat, target_score, args)
        xai_metrics["faithfulness"] = faith
    return xai_metrics


def process_image(model, names, image_path, out_dir, args):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skip unreadable image: {image_path}")
        return None

    h, w = image.shape[:2]
    gts = read_yolo_labels(image_to_label_path(image_path), w, h, names)
    preds = predict_boxes(model, image, names, args.imgsz, args.conf, args.iou, args.device)
    if args.target_class is not None:
        target_name = str(args.target_class)
        preds = [p for p in preds if str(p.cls) == target_name or p.name == target_name]
    if not preds:
        det_eval = no_prediction_evaluation(gts, args)
        print(f"No prediction ({det_eval['case_type']}): {image_path}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = save_no_prediction_outputs(image, out_dir, image_path.stem, gts)
        analysis = {
            "image": str(image_path),
            "knowledge_file": str(args.knowledge),
            "xai_method": getattr(args, "xai_method", "both"),
            "target": {"cls": None, "name": "No prediction", "conf": 0.0, "xyxy": []},
            "defect_knowledge": {},
            "detections": [],
            "num_explained": 0,
            "num_predictions": 0,
            "num_ground_truth": len(gts),
            "ground_truth": [asdict(gt) for gt in gts],
            "matched_gt": None,
            "matched_gt_iou": 0.0,
            "detection_evaluation": det_eval,
            "detection_summary": detection_summary(preds, gts, args),
            "heat_in_pred_bbox_ratio": None,
            "heat_in_gt_bbox_ratio": None,
            "occlusion_heat_in_pred_bbox_ratio": None,
            "occlusion_heat_in_gt_bbox_ratio": None,
            "gradcam_heat_in_pred_bbox_ratio": None,
            "gradcam_heat_in_gt_bbox_ratio": None,
            "xai_runtime_sec": {},
            "xai_comparison": [],
            "xai_metrics": {"skipped": "no predicted bbox target"},
            "outputs": {k: str(v) for k, v in paths.items()},
            "llm_answer": None,
        }
        add_scientific_metadata(analysis, None, args, skipped=True)
        analysis["customer_metadata"] = customer_metadata(analysis)
        json_path = out_dir / f"{image_path.stem}_xai.json"
        md_path = out_dir / f"{image_path.stem}_report.md"
        json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(deterministic_report(analysis, args.audience) + "\n", encoding="utf-8")
        print(f"Saved: {paths['overlay']}")
        return analysis

    if getattr(args, "explain_all", False):
        return process_image_multi(model, names, image_path, image, preds, out_dir, args)

    target = preds[min(args.target_index, len(preds) - 1)]
    gt, gt_iou, det_eval = detection_evaluation_for_target(target, gts, args)

    print(f"XAI: {image_path.name} -> {target.name} {target.conf:.3f}", flush=True)
    run_occ = xai_method_enabled(args, "occlusion")
    run_gc = xai_method_enabled(args, "gradcam")
    runtimes = {}

    heat = None
    if run_occ:
        start = time.perf_counter()
        heat = occlusion_heatmap(model, image, names, target, args)
        runtimes["occlusion"] = time.perf_counter() - start

    heat_gc = None
    if run_gc:
        start = time.perf_counter()
        try:
            heat_gc = gradcam_heatmap(model, image, target)
            runtimes["gradcam"] = time.perf_counter() - start
            print(f"  Grad-CAM: done", flush=True)
        except Exception as e:
            runtimes["gradcam"] = time.perf_counter() - start
            import traceback
            print(f"  Grad-CAM skipped: {e}\n{traceback.format_exc()}", flush=True)

    selected_heat = heat if heat is not None else heat_gc
    if selected_heat is None:
        selected_heat = np.zeros((h, w), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = save_heatmap_outputs(image, selected_heat, out_dir, image_path.stem, preds, gts, target)

    gc_paths = {}
    if heat_gc is not None and getattr(args, "xai_method", "both") == "both":
        gc_paths = save_gradcam_outputs(image, heat_gc, out_dir, image_path.stem, preds, [target])

    occ_pred_ratio = heat_ratio(heat, target.xyxy)
    occ_gt_ratio = heat_ratio(heat, gt.xyxy) if (heat is not None and gt) else None
    gc_pred_ratio = heat_ratio(heat_gc, target.xyxy)
    gc_gt_ratio = heat_ratio(heat_gc, gt.xyxy) if (heat_gc is not None and gt) else None
    selected_pred_ratio = heat_ratio(selected_heat, target.xyxy)
    selected_gt_ratio = heat_ratio(selected_heat, gt.xyxy) if gt else None

    analysis = {
        "image": str(image_path),
        "knowledge_file": str(args.knowledge),
        "xai_method": getattr(args, "xai_method", "both"),
        "target": asdict(target),
        "defect_knowledge": knowledge_for_class(getattr(args, "knowledge_data", {}), target.cls, target.name),
        "detections": [
            {
                "target": asdict(pred),
                "defect_knowledge": knowledge_for_class(getattr(args, "knowledge_data", {}), pred.cls, pred.name),
            }
            for pred in preds
        ],
        "num_explained": 1,
        "num_predictions": len(preds),
        "num_ground_truth": len(gts),
        "ground_truth": [asdict(gt_item) for gt_item in gts],
        "matched_gt": asdict(gt) if gt else None,
        "matched_gt_iou": float(gt_iou),
        "detection_evaluation": det_eval,
        "detection_summary": detection_summary(preds, gts, args),
        "heat_in_pred_bbox_ratio": selected_pred_ratio,
        "heat_in_gt_bbox_ratio": selected_gt_ratio,
        "occlusion_heat_in_pred_bbox_ratio": occ_pred_ratio,
        "occlusion_heat_in_gt_bbox_ratio": occ_gt_ratio,
        "gradcam_heat_in_pred_bbox_ratio": gc_pred_ratio,
        "gradcam_heat_in_gt_bbox_ratio": gc_gt_ratio,
        "xai_runtime_sec": {k: round(float(v), 3) for k, v in runtimes.items()},
        "xai_comparison": xai_comparison_from_heats(heat, heat_gc, target, gt, w, h, runtimes),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    add_scientific_metadata(analysis, target, args)

    if heat_gc is not None:
        analysis["outputs"].update({k: str(v) for k, v in gc_paths.items()})

    xai_metrics = compute_xai_metrics(
        model, image, names, target, args, heat, heat_gc, gt, w, h
    )
    analysis["xai_metrics"] = xai_metrics

    llm_paths = {**paths, **gc_paths}
    llm_answer = ask_vision_llm(args, llm_paths, analysis) if args.llm else None
    analysis["customer_metadata"] = customer_metadata(analysis)
    analysis["llm_answer"] = llm_answer

    json_path = out_dir / f"{image_path.stem}_xai.json"
    md_path = out_dir / f"{image_path.stem}_report.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    if llm_answer and args.audience == "customer":
        report = "Câu trả lời từ LLM:\n" + llm_answer
    else:
        report = deterministic_report(analysis, args.audience)
        comp_rep = comparison_report(analysis)
        if comp_rep:
            report += f"\n\n{comp_rep}"
        if llm_answer:
            title = "LLM/VLM answer"
            report += f"\n\n{title}:\n" + llm_answer
    md_path.write_text(report + "\n", encoding="utf-8")
    print(f"Saved: {paths['overlay']}")
    return analysis


def process_image_multi(model, names, image_path, image, preds, out_dir, args):
    h, w = image.shape[:2]
    gts = read_yolo_labels(image_to_label_path(image_path), w, h, names)
    targets = preds[: max(1, int(getattr(args, "max_targets", 5)))]
    selected_heatmaps = []
    occ_heatmaps = []
    gc_heatmaps = []
    detections = []
    runtimes = {"occlusion": 0.0, "gradcam": 0.0}
    run_occ = xai_method_enabled(args, "occlusion")
    run_gc = xai_method_enabled(args, "gradcam")

    print(f"XAI all: {image_path.name} -> {len(targets)} targets", flush=True)
    for idx, target in enumerate(targets, start=1):
        print(f"  target {idx}/{len(targets)}: {target.name} {target.conf:.3f}", flush=True)
        heat = None
        if run_occ:
            start = time.perf_counter()
            heat = occlusion_heatmap(model, image, names, target, args)
            runtimes["occlusion"] += time.perf_counter() - start
            occ_heatmaps.append(heat)

        heat_gc = None
        if run_gc:
            start = time.perf_counter()
            try:
                heat_gc = gradcam_heatmap(model, image, target)
                runtimes["gradcam"] += time.perf_counter() - start
                gc_heatmaps.append(heat_gc)
            except Exception as e:
                runtimes["gradcam"] += time.perf_counter() - start
                if getattr(args, "progress", False):
                    print(f"  Grad-CAM skipped: {e}", flush=True)

        selected_heat = heat if heat is not None else heat_gc
        if selected_heat is not None:
            selected_heatmaps.append(selected_heat)

        gt, gt_iou, det_eval = detection_evaluation_for_target(target, gts, args)
        xai_metrics = compute_xai_metrics(
            model, image, names, target, args, heat, heat_gc, gt, w, h,
            prefix=f"  target {idx} "
        )
        det_data = {
            "target": asdict(target),
            "defect_knowledge": knowledge_for_class(getattr(args, "knowledge_data", {}), target.cls, target.name),
            "matched_gt": asdict(gt) if gt else None,
            "matched_gt_iou": float(gt_iou),
            "detection_evaluation": det_eval,
            "heat_in_pred_bbox_ratio": heat_ratio(selected_heat, target.xyxy),
            "heat_in_gt_bbox_ratio": heat_ratio(selected_heat, gt.xyxy) if (selected_heat is not None and gt) else None,
            "occlusion_heat_in_pred_bbox_ratio": heat_ratio(heat, target.xyxy),
            "occlusion_heat_in_gt_bbox_ratio": heat_ratio(heat, gt.xyxy) if (heat is not None and gt) else None,
            "xai_metrics": xai_metrics,
        }
        if heat_gc is not None:
            det_data["gradcam_heat_in_pred_bbox_ratio"] = heat_ratio(heat_gc, target.xyxy)
            if gt:
                det_data["gradcam_heat_in_gt_bbox_ratio"] = heat_ratio(heat_gc, gt.xyxy)
        detections.append(det_data)

    combined = combine_heatmaps(selected_heatmaps, h, w)
    if combined is None:
        combined = np.zeros((h, w), dtype=np.float32)
    combined_occ = combine_heatmaps(occ_heatmaps, h, w)
    combined_gc = combine_heatmaps(gc_heatmaps, h, w)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = save_multi_heatmap_outputs(image, combined, out_dir, image_path.stem, preds, gts, targets)

    gc_paths = {}
    if combined_gc is not None and getattr(args, "xai_method", "both") == "both":
        gc_paths = save_gradcam_outputs(image, combined_gc, out_dir, image_path.stem, preds, targets)

    primary = detections[0]
    conf_weights = [max(float(det["target"].get("conf", 0.0)), 0.0) for det in detections]
    weight_sum = sum(conf_weights)
    def weighted_ratio(key, fallback=None):
        valid = [
            (det.get(key), weight)
            for det, weight in zip(detections, conf_weights)
            if det.get(key) is not None
        ]
        total = sum(weight for _, weight in valid)
        if total <= 0:
            return fallback
        return sum(value * weight for value, weight in valid) / total

    combined_pred_ratio = weighted_ratio("heat_in_pred_bbox_ratio", primary["heat_in_pred_bbox_ratio"])
    combined_gt_ratio = weighted_ratio("heat_in_gt_bbox_ratio", primary["heat_in_gt_bbox_ratio"])
    combined_occ_pred_ratio = weighted_ratio("occlusion_heat_in_pred_bbox_ratio")
    combined_occ_gt_ratio = weighted_ratio("occlusion_heat_in_gt_bbox_ratio")
    combined_gc_pred_ratio = weighted_ratio("gradcam_heat_in_pred_bbox_ratio")
    combined_gc_gt_ratio = weighted_ratio("gradcam_heat_in_gt_bbox_ratio")
    xai_rows = []
    if combined_occ_pred_ratio is not None:
        xai_rows.append(xai_method_summary(
            "Occlusion", runtimes.get("occlusion", 0.0),
            combined_occ_pred_ratio, combined_occ_gt_ratio
        ))
    if combined_gc_pred_ratio is not None:
        xai_rows.append(xai_method_summary(
            "Grad-CAM", runtimes.get("gradcam", 0.0),
            combined_gc_pred_ratio, combined_gc_gt_ratio
        ))

    analysis = {
        "image": str(image_path),
        "knowledge_file": str(args.knowledge),
        "xai_method": getattr(args, "xai_method", "both"),
        "target": primary["target"],
        "defect_knowledge": primary["defect_knowledge"],
        "detections": detections,
        "num_explained": len(targets),
        "num_predictions": len(preds),
        "num_ground_truth": len(gts),
        "ground_truth": [asdict(gt_item) for gt_item in gts],
        "matched_gt": primary["matched_gt"],
        "matched_gt_iou": primary["matched_gt_iou"],
        "detection_evaluation": primary["detection_evaluation"],
        "detection_summary": detection_summary(preds, gts, args),
        "heat_in_pred_bbox_ratio": combined_pred_ratio,
        "heat_in_gt_bbox_ratio": combined_gt_ratio,
        "occlusion_heat_in_pred_bbox_ratio": combined_occ_pred_ratio,
        "occlusion_heat_in_gt_bbox_ratio": combined_occ_gt_ratio,
        "gradcam_heat_in_pred_bbox_ratio": combined_gc_pred_ratio,
        "gradcam_heat_in_gt_bbox_ratio": combined_gc_gt_ratio,
        "xai_runtime_sec": {k: round(float(v), 3) for k, v in runtimes.items() if v > 0},
        "xai_comparison": rank_xai_rows(xai_rows),
        "xai_metrics": primary.get("xai_metrics", {}),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    add_scientific_metadata(analysis, DetBox(**{
        "cls": primary["target"]["cls"],
        "name": primary["target"]["name"],
        "conf": primary["target"]["conf"],
        "xyxy": primary["target"]["xyxy"],
    }), args)

    if combined_gc is not None:
        analysis["outputs"].update({k: str(v) for k, v in gc_paths.items()})

    llm_paths = {**paths, **gc_paths}
    llm_answer = ask_vision_llm(args, llm_paths, analysis) if args.llm else None
    analysis["customer_metadata"] = customer_metadata(analysis)
    analysis["llm_answer"] = llm_answer

    json_path = out_dir / f"{image_path.stem}_xai.json"
    md_path = out_dir / f"{image_path.stem}_report.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    if llm_answer and args.audience == "customer":
        report = "Câu trả lời từ LLM:\n" + llm_answer
    else:
        report = deterministic_report(analysis, args.audience)
        comp_rep = comparison_report(analysis)
        if comp_rep:
            report += f"\n\n{comp_rep}"
        if llm_answer:
            title = "LLM/VLM answer"
            report += f"\n\n{title}:\n" + llm_answer
    md_path.write_text(report + "\n", encoding="utf-8")
    print(f"Saved: {paths['overlay']}")
    return analysis


def collect_images(args):
    if args.image:
        return [Path(args.image)]
    source = Path(args.source_dir)
    images = [p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(images)[: args.max_images]


def output_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "code":
        return PROJECT_ROOT / path
    return XAI_DIR / path


def parse_args():
    p = argparse.ArgumentParser(description="YOLO XAI heatmap + optional vision LLM report")
    p.add_argument("--weights", default="model/best.pt")
    p.add_argument("--data", default="voc2007_yolo/dataset.yaml")
    p.add_argument("--image")
    p.add_argument("--source-dir", default="voc2007_yolo/images/test")
    p.add_argument("--out", default=str(XAI_DIR / "xai_results" / "yolo_occlusion_llm"))
    p.add_argument("--max-images", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default=None)
    p.add_argument("--patch", type=int, default=120)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--fill-mode", choices=["gray", "mean"], default="gray")
    p.add_argument("--xai-method", choices=["occlusion", "gradcam", "both"], default="both")
    p.add_argument("--xai-scope", choices=["full", "bbox"], default="full")
    p.add_argument("--roi-scale", type=float, default=3.0)
    p.add_argument("--min-roi", type=int, default=240)
    p.add_argument("--match-iou", type=float, default=0.30,
                   help="IoU threshold for matching occluded prediction back to target")
    p.add_argument("--eval-iou", type=float, default=0.50,
                   help="IoU threshold for TP/FP/FN/TN detection evaluation")
    p.add_argument("--target-class", default=None, help="class id or class name")
    p.add_argument("--target-index", type=int, default=0, help="0 = highest confidence prediction")
    p.add_argument("--explain-all", action="store_true")
    p.add_argument("--max-targets", type=int, default=5)
    p.add_argument("--knowledge", default="Code/xai/defect_knowledge.yaml")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--eval-metrics", action="store_true",
                   help="Tinh Deletion/Insertion AUC va Faithfulness (cham hon)")
    p.add_argument("--faithfulness-threshold", type=float, default=0.15,
                   help="Nguong drop confidence de dat faithfulness, vi du 0.15 = giam 15%%")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--audience", choices=["customer", "engineer"], default="customer")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--llm-timeout", type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    data_yaml = Path(args.data)
    if not weights.exists():
        print(f"Missing weights: {weights}")
        print("Put trained YOLO weights at model/best.pt or pass --weights runs/detect/train/weights/best.pt")
        return 2
    if not data_yaml.exists():
        print(f"Missing data yaml: {data_yaml}")
        return 2

    names = load_names(data_yaml)
    args.knowledge_data = load_defect_knowledge(args.knowledge)
    images = collect_images(args)
    if not images:
        print("No images found.")
        return 2

    model = YOLO(str(weights))
    out_dir = output_path(args.out)
    all_results = []
    for image_path in images:
        result = process_image(model, names, image_path, out_dir / image_path.stem, args)
        if result:
            all_results.append(result)

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
