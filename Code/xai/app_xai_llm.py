import argparse
import base64
import html
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from flask import Flask, has_request_context, request, send_file, url_for
from ultralytics import YOLO
from werkzeug.exceptions import RequestEntityTooLarge

from yolo_xai_llm import (
    DetBox,
    customer_metadata,
    deterministic_report,
    load_defect_knowledge,
    load_names,
    scan_grid_for_target,
    bbox_iou,
    clamp_box,
    draw_boxes,
    save_no_prediction_outputs,
    knowledge_for_class,
    ask_vision_llm,
    image_to_label_path,
    read_yolo_labels,
    no_prediction_evaluation,
    detection_summary,
    detection_evaluation_for_target,
    heat_ratio,
    add_scientific_metadata,
)
from xai_metrics import (
    pointing_game,
    deletion_auc,
    insertion_auc,
    faithfulness_test,
    comparison_report,
)

ROOT = Path(__file__).resolve().parents[2]
XAI_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "voc2007_yolo" / "dataset.yaml"
DEFAULT_KNOWLEDGE = ROOT / "Code" / "xai" / "defect_knowledge.yaml"
DEFAULT_OUT = XAI_DIR / "xai_results" / "app"
UPLOAD_DIR = DEFAULT_OUT / "uploads"
MAX_UPLOAD_MB = int(os.getenv("XAI_MAX_UPLOAD_MB", "64"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
LLM_SETTINGS = {
    "use_llm": False,
    "llm_base_url": os.getenv("NINEROUTER_URL", "http://localhost:20128"),
    "llm_api_key": os.getenv("NINEROUTER_KEY", ""),
    "llm_model": os.getenv("XAI_LLM_MODEL", ""),
}
PROCESS_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["MAX_FORM_MEMORY_SIZE"] = MAX_UPLOAD_BYTES


@lru_cache(maxsize=1)
def get_ensemble_models():
    """
    Đọc cấu hình từ ensemble_config.json và khởi tạo 3 mô hình con trên GPU.
    """
    config_path = XAI_DIR / "conf" / "ensemble_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware Setup] Phát hiện thiết bị: {device.upper()}")

    # Đọc cấu hình đường dẫn từ JSON, fallback về các file mặc định nếu không khai báo
    raw_paths = cfg.get("model_paths", {
        "kq9": "model/ensemple/best9.pt",
        "kq11": "model/ensemple/best11.pt",
        "kq2": "model/ensemple/best2.pt"
    })

    model_paths = {}
    for name, p in raw_paths.items():
        p_path = Path(p)
        if not p_path.is_absolute():
            model_paths[name] = XAI_DIR / p_path
        else:
            model_paths[name] = p_path

    models = {}
    for name in cfg["model_order"]:
        path = model_paths[name]
        print(f"  -> Khởi tạo mô hình {name} trên GPU: {path}")
        m = YOLO(str(path))
        m.to(device)
        models[name] = m

    return models, cfg


@lru_cache(maxsize=1)
def get_knowledge(path):
    return load_defect_knowledge(path)


def normalize_base_url(url):
    url = (url or "").strip()
    if not url:
        return None
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def save_llm_settings(form):
    LLM_SETTINGS["use_llm"] = form.get("use_llm") == "on"
    for key in ("llm_base_url", "llm_api_key", "llm_model"):
        value = (form.get(key) or "").strip()
        if value:
            LLM_SETTINGS[key] = value


def llm_setting(key):
    return LLM_SETTINGS.get(key, "")


def rel_url(path):
    if not path:
        return ""
    rel = Path(path).resolve().relative_to(ROOT)
    if not has_request_context():
        return "/files/" + rel.as_posix()
    return url_for("serve_file", relpath=rel.as_posix())


def retained_image_token(path):
    try:
        resolved = Path(path).resolve()
        if UPLOAD_DIR.resolve() not in resolved.parents:
            return ""
        return resolved.relative_to(DEFAULT_OUT.resolve()).as_posix()
    except ValueError:
        return ""


def retained_image_path(token):
    if not token:
        return None
    candidate = (DEFAULT_OUT / token).resolve()
    upload_root = UPLOAD_DIR.resolve()
    if upload_root not in candidate.parents:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def safe_file_name(name):
    keep = []
    for ch in Path(name).name:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep) or "upload.jpg"


def save_camera_image(data_url):
    if not data_url or "," not in data_url:
        return None
    header, encoded = data_url.split(",", 1)
    if "base64" not in header:
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_camera.jpg"
    file_path.write_bytes(base64.b64decode(encoded))
    return file_path


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    return page({
        "ok": False,
        "message": (
            f"Ảnh gửi lên quá lớn. Giới hạn hiện tại là {MAX_UPLOAD_MB} MB. "
            "Hãy giảm kích thước ảnh, dùng ảnh JPG, hoặc tăng biến XAI_MAX_UPLOAD_MB."
        ),
    }), 413


def ensemble_predict_batch_wrapper(images, models, cfg, device="cuda"):
    """
    Black-box Wrapper cho Ensemble hỗ trợ BATCH và GPU.
    QUAN TRỌNG: Truyền ảnh numpy BGR gốc trực tiếp vào model.predict() —
    KHÔNG được tiền xử lý thành tensor vì YOLO tự xử lý bên trong.
    Tọa độ trả về ở dạng chuẩn hóa [0, 1] dựa trên kích thước ảnh thực tế.
    """
    imgsz = cfg.get("imgsz", 960)
    pred_conf = cfg.get("pred_conf", 0.05)
    pred_iou = cfg.get("pred_iou", 0.5)
    max_det = cfg.get("max_det", 1000)
    wbf_iou_thr = cfg.get("wbf_iou_thr", 0.55)
    final_nms_iou = cfg.get("final_nms_iou", 0.6)
    use_final_nms = cfg.get("use_final_nms", True)

    if not torch.cuda.is_available():
        device = "cpu"

    # Chấp nhận list of numpy arrays (BGR) — đây là format chuẩn
    if isinstance(images, list):
        imgs_list = images
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        imgs_list = [images]
    else:
        # Trường hợp hiếm: tensor → chuyển lại numpy
        imgs_list = []
        t = images if isinstance(images, torch.Tensor) else torch.tensor(images)
        for i in range(t.shape[0]):
            img_np = (t[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            imgs_list.append(img_np)

    B = len(imgs_list)
    model_order = cfg["model_order"]

    # Lấy kích thước ảnh thực tế từ ảnh đầu tiên (dùng để chuẩn hóa tọa độ cho WBF)
    img_h, img_w = imgs_list[0].shape[:2]

    # Cấu trúc lưu trữ dự đoán thô của từng mô hình cho mỗi ảnh trong batch
    all_boxes = [[] for _ in range(B)]
    all_scores = [[] for _ in range(B)]
    all_labels = [[] for _ in range(B)]

    # Chia nhỏ mini-batch để tránh tràn bộ nhớ GPU (OOM) trên card 4GB VRAM
    mini_batch_size = 4

    with torch.no_grad():
        for start_idx in range(0, B, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, B)
            # Truyền trực tiếp list ảnh numpy BGR — YOLO tự resize và tiền xử lý
            mini_batch_imgs = imgs_list[start_idx:end_idx]

            for name in model_order:
                model = models[name]

                # Truyền ảnh numpy gốc, YOLO tự xử lý toàn bộ trên GPU
                results = model.predict(
                    mini_batch_imgs,
                    imgsz=imgsz,
                    conf=pred_conf,
                    iou=pred_iou,
                    max_det=max_det,
                    device=device,
                    verbose=False,
                    half=(device == "cuda"),  # FP16 trên GPU để tăng tốc và tiết kiệm VRAM
                )

                for local_i, res in enumerate(results):
                    global_i = start_idx + local_i
                    if res.boxes is not None and len(res.boxes) > 0:
                        # YOLO trả về tọa độ xyxy trong hệ tọa độ ảnh gốc
                        xyxy = res.boxes.xyxy.cpu().numpy()

                        # Chuẩn hóa về [0, 1] bằng kích thước ảnh THỰC TẾ (không phải imgsz)
                        boxes_norm = np.zeros_like(xyxy, dtype=np.float32)
                        boxes_norm[:, 0] = xyxy[:, 0] / img_w  # x1
                        boxes_norm[:, 1] = xyxy[:, 1] / img_h  # y1
                        boxes_norm[:, 2] = xyxy[:, 2] / img_w  # x2
                        boxes_norm[:, 3] = xyxy[:, 3] / img_h  # y2
                        boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

                        scores_cpu = res.boxes.conf.cpu().numpy()
                        labels_cpu = res.boxes.cls.cpu().numpy().astype(int)

                        all_boxes[global_i].append(boxes_norm)
                        all_scores[global_i].append(scores_cpu)
                        all_labels[global_i].append(labels_cpu)
                    else:
                        all_boxes[global_i].append(np.empty((0, 4), dtype=np.float32))
                        all_scores[global_i].append(np.empty((0,), dtype=np.float32))
                        all_labels[global_i].append(np.empty((0,), dtype=np.int32))

            # Giải phóng bộ nhớ GPU sau mỗi mini-batch
            if device == "cuda":
                torch.cuda.empty_cache()

    # Áp dụng thuật toán gộp WBF và lọc NMS cho từng ảnh trong batch
    final_results = []
    for i in range(B):
        boxes_list = all_boxes[i]
        scores_list = all_scores[i]
        labels_list = all_labels[i]

        has_any = any(len(b) > 0 for b in boxes_list)
        if not has_any:
            final_results.append({
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32)
            })
            continue

        unique_labels = set()
        for labels in labels_list:
            unique_labels.update(labels)

        merged_boxes = []
        merged_scores = []
        merged_labels = []

        # WBF lớp theo lớp để áp dụng class_model_weights
        for class_id in sorted(unique_labels):
            cls_boxes_list = []
            cls_scores_list = []
            cls_labels_list = []

            for m_idx in range(len(model_order)):
                mask = (labels_list[m_idx] == class_id)
                cls_boxes_list.append(boxes_list[m_idx][mask])
                cls_scores_list.append(scores_list[m_idx][mask])
                cls_labels_list.append(labels_list[m_idx][mask])

            class_str = str(class_id)
            if "class_model_weights" in cfg and class_str in cfg["class_model_weights"]:
                weights = cfg["class_model_weights"][class_str]
            else:
                weights = [cfg["global_model_weights"][name] for name in model_order]

            if sum(len(b) for b in cls_boxes_list) == 0:
                continue

            cls_wbf_boxes, cls_wbf_scores, cls_wbf_labels = weighted_boxes_fusion(
                cls_boxes_list,
                cls_scores_list,
                cls_labels_list,
                weights=weights,
                iou_thr=wbf_iou_thr,
                skip_box_thr=pred_conf
            )

            merged_boxes.extend(cls_wbf_boxes)
            merged_scores.extend(cls_wbf_scores)
            merged_labels.extend(cls_wbf_labels)

        if not merged_boxes:
            final_results.append({
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32)
            })
            continue

        merged_boxes = np.array(merged_boxes)
        merged_scores = np.array(merged_scores)
        merged_labels = np.array(merged_labels).astype(int)

        # Lọc NMS cuối cùng nếu cấu hình yêu cầu
        if use_final_nms and len(merged_boxes) > 1:
            nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in merged_boxes]
            indices = cv2.dnn.NMSBoxes(
                bboxes=nms_boxes,
                scores=merged_scores.tolist(),
                score_threshold=pred_conf,
                nms_threshold=final_nms_iou
            )
            if len(indices) > 0:
                indices = indices.flatten()
                merged_boxes = merged_boxes[indices]
                merged_scores = merged_scores[indices]
                merged_labels = merged_labels[indices]
            else:
                merged_boxes = np.empty((0, 4), dtype=np.float32)
                merged_scores = np.empty((0,), dtype=np.float32)
                merged_labels = np.empty((0,), dtype=np.int32)

        # Lọc lại kết quả cuối cùng theo post_conf cụ thể cho từng Class
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for b, s, l in zip(merged_boxes, merged_scores, merged_labels):
            class_str = str(l)
            post_conf_thr = cfg.get("post_conf", {}).get(class_str, pred_conf)
            if s >= post_conf_thr:
                filtered_boxes.append(b)
                filtered_scores.append(s)
                filtered_labels.append(l)

        final_results.append({
            "boxes": np.array(filtered_boxes) if filtered_boxes else np.empty((0, 4), dtype=np.float32),
            "scores": np.array(filtered_scores) if filtered_scores else np.empty((0,), dtype=np.float32),
            "labels": np.array(filtered_labels).astype(int) if filtered_labels else np.empty((0,), dtype=np.int32)
        })

    return final_results


def ensemble_occlusion_heatmap(models, cfg, image, target, args, device="cuda"):
    """
    Sinh bản đồ nhiệt bằng Occlusion chạy BATCH trên GPU.
    Cải tiến:
    - Patch/stride tự động theo kích thước bbox (nhỏ hơn = chi tiết hơn).
    - Soft Gaussian mask tập trung vào vùng bbox.
    - Gaussian smoothing + lọc nhiễu.
    """
    h, w = image.shape[:2]

    # === Adaptive patch/stride theo kích thước bbox ===
    x1t, y1t, x2t, y2t = target.xyxy
    bw = max(1.0, x2t - x1t)
    bh = max(1.0, y2t - y1t)
    bbox_min_dim = min(bw, bh)

    # Patch tỉ lệ ~50% kích thước nhỏ nhất bbox (tối thiểu 60px, tối đa patch gốc)
    adaptive_patch = max(60, min(int(bbox_min_dim * 0.5), args.patch))
    adaptive_stride = max(30, int(adaptive_patch * 0.65))

    # Ghi đè tạm args để scan_grid_for_target dùng adaptive values
    orig_patch = args.patch
    orig_stride = args.stride
    args.patch = adaptive_patch
    args.stride = adaptive_stride

    y_positions, x_positions = scan_grid_for_target(image, target, args)

    # Khôi phục args gốc
    args.patch = orig_patch
    args.stride = orig_stride

    total_patches = len(y_positions) * len(x_positions)
    print(f"  Occlusion: {total_patches} patches ({len(x_positions)}x{len(y_positions)}), "
          f"patch={adaptive_patch}, stride={adaptive_stride}, scope={getattr(args, 'xai_scope', 'bbox')}",
          flush=True)

    perturbed_images = []
    positions = []

    if getattr(args, "fill_mode", "gray") == "mean":
        fill = image.mean(axis=(0, 1)).astype(np.uint8)
    else:
        fill = np.array([127, 127, 127], dtype=np.uint8)

    for y in y_positions:
        for x in x_positions:
            y2 = min(h, y + adaptive_patch)
            x2 = min(w, x + adaptive_patch)
            occluded = image.copy()
            occluded[y:y2, x:x2] = fill
            perturbed_images.append(occluded)
            positions.append((y, y2, x, x2))

    # Chạy inference song song cho toàn bộ batch trên GPU
    batch_results = ensemble_predict_batch_wrapper(perturbed_images, models, cfg, device)

    match_iou = float(getattr(args, "match_iou", 0.30))
    base_score = max(target.conf, 1e-8)

    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for idx, res in enumerate(batch_results):
        y, y2, x, x2 = positions[idx]

        new_score = 0.0
        if len(res["boxes"]) > 0:
            scores_matched = []
            for b, s, l in zip(res["boxes"], res["scores"], res["labels"]):
                if l == target.cls:
                    bx1_abs = b[0] * w
                    by1_abs = b[1] * h
                    bx2_abs = b[2] * w
                    by2_abs = b[3] * h

                    iou = bbox_iou([bx1_abs, by1_abs, bx2_abs, by2_abs], target.xyxy)
                    if iou >= match_iou:
                        scores_matched.append(s)
            if scores_matched:
                new_score = max(scores_matched)

        drop = max(0.0, base_score - new_score)
        heat[y:y2, x:x2] += drop
        counts[y:y2, x:x2] += 1.0

    heat = heat / np.maximum(counts, 1.0)

    # === Soft Gaussian mask: tập trung heat vào vùng bbox ===
    cx = (x1t + x2t) / 2.0
    cy = (y1t + y2t) / 2.0
    # Sigma = occlusion_roi_scale x bbox size → mặc định 0.8
    roi_scale = cfg.get("occlusion_roi_scale", 0.8)
    sigma_x = max(bw * roi_scale, 20.0)
    sigma_y = max(bh * roi_scale, 20.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bbox_mask = np.exp(
        -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
    )
    heat = heat * bbox_mask

    # Lọc nhiễu: bỏ các vùng drop quá nhỏ (dưới noise_threshold, mặc định 0.08)
    noise_thr_pct = cfg.get("occlusion_noise_threshold", 0.08)
    if heat.max() > 0:
        noise_thr = heat.max() * noise_thr_pct
        heat[heat < noise_thr] = 0.0

    # Gaussian smoothing để heatmap mượt hơn
    blur_k = max(adaptive_stride, adaptive_patch) | 1
    heat = cv2.GaussianBlur(heat, (blur_k, blur_k), sigmaX=blur_k / 3.0)

    if heat.max() > 0:
        heat = heat / heat.max()

    return heat


def ensemble_gradcam_heatmap(models, cfg, image, target):
    """
    Sinh bản đồ nhiệt Grad-CAM bằng cách chạy trên từng mô hình con và gộp lại.
    Cải tiến:
    - Chuyển model về FP32 trước backward pass (FP16 không tương thích Grad-CAM).
    - Mask ROI mềm (soft elliptical) quanh vùng detection để loại bỏ nhiễu xa (từ cấu hình).
    - Gaussian smoothing + lọc nhiễu (từ cấu hình).
    """
    from xai_metrics import gradcam_heatmap

    cams = []
    model_order = cfg["model_order"]
    layer_index = cfg.get("gradcam_layer_index", 9)

    for name in model_order:
        model = models[name]
        try:
            # Chuyển model về FP32 trước khi chạy Grad-CAM (backward cần FP32)
            model.float()
            cam = gradcam_heatmap(model, image, target, layer_index=layer_index, imgsz=cfg.get("imgsz", 960))
            cams.append(cam)
            print(f"  [Grad-CAM] {name} (layer {layer_index}): done", flush=True)
        except Exception as e:
            print(f"  [Grad-CAM] Bỏ qua mô hình {name}: {e}")
        finally:
            # Chuyển lại FP16 để tiết kiệm VRAM cho các lần predict tiếp theo
            if torch.cuda.is_available():
                model.half()

    if not cams:
        return None

    # Gộp các heatmap của từng mô hình bằng giá trị Max
    combined = np.maximum.reduce(cams)

    h, w = image.shape[:2]

    # === ROI Soft Mask: tập trung Grad-CAM vào vùng quanh detection ===
    x1, y1, x2, y2 = target.xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Sigma = gradcam_roi_scale x bbox → mặc định 0.7
    roi_scale = cfg.get("gradcam_roi_scale", 0.7)
    sigma_x = max(bw * roi_scale, 15.0)
    sigma_y = max(bh * roi_scale, 15.0)

    # Tạo Gaussian mask 2D mềm — trung tâm tại bbox, giảm dần ra xa
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gaussian_mask = np.exp(
        -0.5 * (((xx - cx) / max(sigma_x, 1.0)) ** 2 + ((yy - cy) / max(sigma_y, 1.0)) ** 2)
    )
    # Áp mask lên heatmap
    combined = combined * gaussian_mask

    # Lọc nhiễu: bỏ vùng activation < noise_threshold so với max (mặc định 0.15)
    noise_thr_pct = cfg.get("gradcam_noise_threshold", 0.15)
    if combined.max() > 0:
        noise_thr = combined.max() * noise_thr_pct
        combined[combined < noise_thr] = 0.0

    # Gaussian smoothing để mượt hơn
    blur_k = max(15, int(min(bw, bh) * 0.3)) | 1  # Kernel tỉ lệ với bbox
    combined = cv2.GaussianBlur(combined, (blur_k, blur_k), sigmaX=blur_k / 3.0)

    if combined.max() > 0:
        combined = combined / combined.max()
    return combined


def ensemble_target_score(models, cfg, image, names, target, args):
    """
    Tính score của target trên ảnh biến thể để phục vụ việc đánh giá Deletion/Insertion AUC.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = ensemble_predict_batch_wrapper([image], models, cfg, device)[0]

    h, w = image.shape[:2]
    match_iou = float(getattr(args, "match_iou", 0.30))

    scores_matched = []
    if len(res["boxes"]) > 0:
        for b, s, l in zip(res["boxes"], res["scores"], res["labels"]):
            if l == target.cls:
                bx1_abs = b[0] * w
                by1_abs = b[1] * h
                bx2_abs = b[2] * w
                by2_abs = b[3] * h
                iou = bbox_iou([bx1_abs, by1_abs, bx2_abs, by2_abs], target.xyxy)
                if iou >= match_iou:
                    scores_matched.append(s)

    return max(scores_matched) if scores_matched else 0.0


def save_ensemble_heatmap_outputs(image, heat, out_dir, stem, preds, gts, target, cfg):
    """
    Kết xuất ảnh kết hợp: Ảnh gốc + Heatmap XAI + Bounding boxes + Class Names từ JSON.
    """
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # Trộn ảnh gốc và heatmap
    overlay = cv2.addWeighted(image, 0.58, heat_color, 0.42, 0)

    h, w = overlay.shape[:2]
    class_names = cfg.get("class_names", {})

    def draw_ensemble_boxes(img, boxes, color, prefix):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = clamp_box(box.xyxy, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # Lấy tên lớp hiển thị từ file JSON config
            cls_name = class_names.get(str(box.cls), box.name)
            label = f"{prefix} {cls_name}"
            if box.conf < 1.0:
                label += f" {box.conf:.2f}"
            cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out

    overlay = draw_ensemble_boxes(overlay, preds, (0, 255, 0), "loi")
    overlay = draw_ensemble_boxes(overlay, [target], (0, 255, 255), "vung chinh")
    pred_img = draw_ensemble_boxes(image, preds, (0, 255, 0), "loi")
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


def save_ensemble_multi_heatmap_outputs(image, heat, out_dir, stem, preds, gts, targets, cfg):
    """
    Kết xuất ảnh kết hợp khi giải thích nhiều mục tiêu lỗi.
    """
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.58, heat_color, 0.42, 0)

    h, w = overlay.shape[:2]
    class_names = cfg.get("class_names", {})

    def draw_ensemble_boxes(img, boxes, color, prefix):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = clamp_box(box.xyxy, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cls_name = class_names.get(str(box.cls), box.name)
            label = f"{prefix} {cls_name}"
            if box.conf < 1.0:
                label += f" {box.conf:.2f}"
            cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out

    overlay = draw_ensemble_boxes(overlay, preds, (0, 255, 0), "loi")
    overlay = draw_ensemble_boxes(overlay, targets, (0, 255, 255), "giai thich")
    pred_img = draw_ensemble_boxes(image, preds, (0, 255, 0), "loi")
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


def save_ensemble_gradcam_outputs(image, heat_gc, out_dir, stem, preds, targets, cfg):
    """
    Kết xuất Grad-CAM kết hợp bounding box cho cụm Ensemble.
    """
    gc_u8 = np.uint8(255 * heat_gc)
    gc_color = cv2.applyColorMap(gc_u8, cv2.COLORMAP_JET)
    gc_overlay = cv2.addWeighted(image, 0.58, gc_color, 0.42, 0)

    h, w = gc_overlay.shape[:2]
    class_names = cfg.get("class_names", {})

    def draw_ensemble_boxes(img, boxes, color, prefix):
        out = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = clamp_box(box.xyxy, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cls_name = class_names.get(str(box.cls), box.name)
            label = f"{prefix} {cls_name}"
            if box.conf < 1.0:
                label += f" {box.conf:.2f}"
            cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out

    gc_overlay = draw_ensemble_boxes(gc_overlay, preds, (0, 255, 0), "loi")
    gc_overlay = draw_ensemble_boxes(gc_overlay, targets, (0, 255, 255), "vung chinh")

    gc_paths = {
        "gradcam": out_dir / f"{stem}_gradcam.jpg",
        "gradcam_overlay": out_dir / f"{stem}_gradcam_overlay.jpg",
    }

    cv2.imwrite(str(gc_paths["gradcam"]), gc_color)
    cv2.imwrite(str(gc_paths["gradcam_overlay"]), gc_overlay)

    return gc_paths


def process_image_ensemble(models, cfg, names, image_path, out_dir, args):
    """
    Quy trình xử lý ảnh chính cho Ensemble và chạy thuật toán giải thích XAI.
    """
    from dataclasses import asdict

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skip unreadable image: {image_path}")
        return None

    h, w = image.shape[:2]
    gts = read_yolo_labels(image_to_label_path(image_path), w, h, names)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chạy dự đoán Ensemble trên ảnh gốc
    orig_preds_dict = ensemble_predict_batch_wrapper([image], models, cfg, device)[0]
    preds = []
    if len(orig_preds_dict["boxes"]) > 0:
        for b, s, l in zip(orig_preds_dict["boxes"], orig_preds_dict["scores"], orig_preds_dict["labels"]):
            x1_abs = b[0] * w
            y1_abs = b[1] * h
            x2_abs = b[2] * w
            y2_abs = b[3] * h
            class_names = cfg.get("class_names", {})
            cls_name = class_names.get(str(l), str(l))
            preds.append(DetBox(
                cls=int(l),
                name=cls_name,
                conf=float(s),
                xyxy=[x1_abs, y1_abs, x2_abs, y2_abs]
            ))

    preds = sorted(preds, key=lambda x: x.conf, reverse=True)

    if args.target_class is not None:
        target_name = str(args.target_class)
        preds = [p for p in preds if str(p.cls) == target_name or p.name == target_name]

    # Xử lý trường hợp không có lỗi được phát hiện
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
        return process_image_ensemble_multi(models, cfg, names, image_path, image, preds, out_dir, args)

    target = preds[min(args.target_index, len(preds) - 1)]
    gt, gt_iou, det_eval = detection_evaluation_for_target(target, gts, args)

    print(f"XAI: {image_path.name} -> {target.name} {target.conf:.3f}", flush=True)

    selected_method = getattr(args, "xai_method", "both")
    run_occ = (selected_method == "both" or selected_method == "occlusion")
    run_gc = (selected_method == "both" or selected_method == "gradcam")
    runtimes = {}

    # Chạy Occlusion trên GPU
    heat = None
    if run_occ:
        start = time.perf_counter()
        heat = ensemble_occlusion_heatmap(models, cfg, image, target, args, device)
        runtimes["occlusion"] = time.perf_counter() - start

    # Chạy Grad-CAM cho Ensemble
    heat_gc = None
    if run_gc:
        start = time.perf_counter()
        try:
            heat_gc = ensemble_gradcam_heatmap(models, cfg, image, target)
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
    paths = save_ensemble_heatmap_outputs(image, selected_heat, out_dir, image_path.stem, preds, gts, target, cfg)

    gc_paths = {}
    if heat_gc is not None and getattr(args, "xai_method", "both") == "both":
        gc_paths = save_ensemble_gradcam_outputs(image, heat_gc, out_dir, image_path.stem, preds, [target], cfg)

    occ_pred_ratio = heat_ratio(heat, target.xyxy)
    occ_gt_ratio = heat_ratio(heat, gt.xyxy) if (heat is not None and gt) else None
    gc_pred_ratio = heat_ratio(heat_gc, target.xyxy)
    gc_gt_ratio = heat_ratio(heat_gc, gt.xyxy) if (heat_gc is not None and gt) else None
    selected_pred_ratio = heat_ratio(selected_heat, target.xyxy)
    selected_gt_ratio = heat_ratio(selected_heat, gt.xyxy) if gt else None

    # Hàm tạo bảng so sánh hiệu năng
    def xai_comparison_from_heats_ensemble(heat_occ, heat_gc, target, gt, w, h, runtimes):
        rows = []
        from yolo_xai_llm import xai_method_summary, rank_xai_rows
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
        "xai_comparison": xai_comparison_from_heats_ensemble(heat, heat_gc, target, gt, w, h, runtimes),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    add_scientific_metadata(analysis, target, args)

    if heat_gc is not None:
        analysis["outputs"].update({k: str(v) for k, v in gc_paths.items()})

    # Đánh giá độ tin cậy bằng Deletion/Insertion AUC sử dụng hàm score ensemble
    def ensemble_score_fn(unused_model, img, names, target, args):
        return ensemble_target_score(models, cfg, img, names, target, args)

    xai_metrics = {}
    if gt and heat is not None:
        xai_metrics["pointing_game_occlusion"] = pointing_game(heat, gt.xyxy, w, h)
    if gt and heat_gc is not None:
        xai_metrics["pointing_game_gradcam"] = pointing_game(heat_gc, gt.xyxy, w, h)
    if getattr(args, "eval_metrics", False) and heat is not None:
        print("  Deletion AUC ...", flush=True)
        del_auc, _ = deletion_auc(None, image, names, target, heat, ensemble_score_fn, args)
        print("  Insertion AUC ...", flush=True)
        ins_auc, _ = insertion_auc(None, image, names, target, heat, ensemble_score_fn, args)
        xai_metrics["deletion_auc"] = del_auc
        xai_metrics["insertion_auc"] = ins_auc
        print("  Faithfulness test ...", flush=True)
        faith = faithfulness_test(None, image, names, target, heat, ensemble_score_fn, args)
        xai_metrics["faithfulness"] = faith

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


def process_image_ensemble_multi(models, cfg, names, image_path, image, preds, out_dir, args):
    """
    Quy trình giải thích nhiều mục tiêu lỗi cùng lúc.
    """
    from dataclasses import asdict
    from yolo_xai_llm import combine_heatmaps

    h, w = image.shape[:2]
    gts = read_yolo_labels(image_to_label_path(image_path), w, h, names)
    targets = preds[: max(1, int(getattr(args, "max_targets", 5)))]

    selected_heatmaps = []
    occ_heatmaps = []
    gc_heatmaps = []
    detections = []
    runtimes = {"occlusion": 0.0, "gradcam": 0.0}

    selected_method = getattr(args, "xai_method", "both")
    run_occ = (selected_method == "both" or selected_method == "occlusion")
    run_gc = (selected_method == "both" or selected_method == "gradcam")

    print(f"XAI all (ensemble): {image_path.name} -> {len(targets)} targets", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, target in enumerate(targets, start=1):
        print(f"  target {idx}/{len(targets)}: {target.name} {target.conf:.3f}", flush=True)
        heat = None
        if run_occ:
            start = time.perf_counter()
            heat = ensemble_occlusion_heatmap(models, cfg, image, target, args, device)
            runtimes["occlusion"] += time.perf_counter() - start
            occ_heatmaps.append(heat)

        heat_gc = None
        if run_gc:
            start = time.perf_counter()
            try:
                heat_gc = ensemble_gradcam_heatmap(models, cfg, image, target)
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

        def ensemble_score_fn(unused_model, img, names, target, args):
            return ensemble_target_score(models, cfg, img, names, target, args)

        xai_metrics = {}
        if gt and heat is not None:
            xai_metrics["pointing_game_occlusion"] = pointing_game(heat, gt.xyxy, w, h)
        if gt and heat_gc is not None:
            xai_metrics["pointing_game_gradcam"] = pointing_game(heat_gc, gt.xyxy, w, h)
        if getattr(args, "eval_metrics", False) and heat is not None:
            del_auc, _ = deletion_auc(None, image, names, target, heat, ensemble_score_fn, args)
            ins_auc, _ = insertion_auc(None, image, names, target, heat, ensemble_score_fn, args)
            xai_metrics["deletion_auc"] = del_auc
            xai_metrics["insertion_auc"] = ins_auc
            faith = faithfulness_test(None, image, names, target, heat, ensemble_score_fn, args)
            xai_metrics["faithfulness"] = faith

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
    paths = save_ensemble_multi_heatmap_outputs(image, combined, out_dir, image_path.stem, preds, gts, targets, cfg)

    gc_paths = {}
    if combined_gc is not None and getattr(args, "xai_method", "both") == "both":
        gc_paths = save_ensemble_gradcam_outputs(image, combined_gc, out_dir, image_path.stem, preds, targets, cfg)

    primary = detections[0]
    conf_weights = [max(float(det["target"].get("conf", 0.0)), 0.0) for det in detections]

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
    from yolo_xai_llm import xai_method_summary, rank_xai_rows
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


def run_pipeline(image_path, form):
    """
    Nhận đầu vào từ Flask form, cấu hình các tham số và chạy quy trình dự đoán & giải thích.
    """
    use_llm = form.get("use_llm") == "on"
    explain_mode = form.get("explain_mode", "all")
    xai_method = form.get("xai_method", "both")

    llm_base_url = (form.get("llm_base_url") or llm_setting("llm_base_url")).strip()
    llm_api_key = (form.get("llm_api_key") or llm_setting("llm_api_key")).strip()
    llm_model = (form.get("llm_model") or llm_setting("llm_model")).strip()

    out_dir = DEFAULT_OUT / f"{time.strftime('%Y%m%d_%H%M%S')}_{Path(image_path).stem}"

    # Đọc cấu hình từ JSON file để khởi tạo
    models, cfg = get_ensemble_models()
    names = {int(k): v for k, v in cfg.get("class_names", {}).items()}

    # Lấy ngưỡng confidence từ form hoặc dùng mặc định từ config file
    conf = float(form.get("conf", str(cfg.get("pred_conf", 0.05))))
    patch = int(form.get("patch", "120"))
    stride = int(form.get("stride", "80"))

    form_values = {
        "conf": f"{conf:g}",
        "patch": str(patch),
        "stride": str(stride),
        "explain_mode": explain_mode,
        "max_targets": form.get("max_targets", "5"),
        "xai_method": xai_method,
    }

    args = SimpleNamespace(
        weights="",
        data=str(DEFAULT_DATA),
        image=str(image_path),
        source_dir="",
        out=str(out_dir),
        max_images=1,
        imgsz=cfg.get("imgsz", 960),  # Đưa imgsz 960 vào predict
        conf=conf,
        iou=cfg.get("pred_iou", 0.5),
        device=None,
        patch=patch,
        stride=stride,
        fill_mode=form.get("fill_mode", "gray"),
        xai_method=xai_method,
        xai_scope=form.get("xai_scope", "bbox"),
        roi_scale=float(form.get("roi_scale", "2.5")),
        min_roi=int(form.get("min_roi", "240")),
        match_iou=0.30,
        eval_iou=0.50,
        target_class=None,
        target_index=0,
        explain_all=explain_mode == "all",
        max_targets=int(form.get("max_targets", "5")),
        knowledge=str(DEFAULT_KNOWLEDGE),
        progress=False,
        eval_metrics=False,
        faithfulness_threshold=0.15,
        llm=use_llm,
        audience="customer",
        llm_model=llm_model or None,
        llm_base_url=normalize_base_url(llm_base_url),
        llm_api_key=llm_api_key or None,
        llm_timeout=120,
        knowledge_data=get_knowledge(str(DEFAULT_KNOWLEDGE)),
    )

    analysis = process_image_ensemble(models, cfg, names, Path(image_path), out_dir, args)
    if not analysis:
        return {
            "ok": False,
            "message": "Không phát hiện lỗi với ngưỡng hiện tại. Thử giảm ngưỡng hoặc dùng ảnh khác.",
            "original_url": rel_url(image_path),
            "retained_image": retained_image_token(image_path),
            "retained_image_name": Path(image_path).name,
            "form_values": form_values,
        }

    outputs = analysis["outputs"]
    report_path = Path(outputs["overlay"]).with_name(f"{Path(image_path).stem}_report.md")
    json_path = Path(outputs["overlay"]).with_name(f"{Path(image_path).stem}_xai.json")
    meta = customer_metadata(analysis)
    detections = analysis.get("detections") or []
    num_explained = int(analysis.get("num_explained") or 0)
    det_eval = analysis.get("detection_evaluation") or {}

    if not detections:
        title = f"Không có prediction - {det_eval.get('case_type', 'FN')}"
    elif len(detections) > 1:
        title = f"{len(detections)} vùng bất thường được phát hiện"
        if num_explained < len(detections):
            title += f" - đang làm nổi bật {num_explained} vùng chính"
    else:
        title = f"{meta['ten_loi']} - mức tin cậy {meta['muc_tin_cay']}"

    report_kind = "rule"
    report_text = deterministic_report(analysis, "customer")
    if use_llm:
        llm_answer = analysis.get("llm_answer")
        if llm_answer:
            report_kind = "llm"
            report_text = llm_answer
        else:
            report_kind = "warn"
            report_text += "\n\nLLM chưa trả lời. Kiểm tra provider vision/image-to-text trong 9Router."

    if analysis.get("xai_method") == "gradcam":
        overlay_label = "Grad-CAM overlay"
        heatmap_label = "Grad-CAM"
    else:
        overlay_label = "Occlusion overlay"
        heatmap_label = "Occlusion heatmap"

    return {
        "ok": True,
        "title": title,
        "report": report_text,
        "report_kind": report_kind,
        "original_url": rel_url(outputs["original"]),
        "pred_url": rel_url(outputs["pred"]),
        "overlay_url": rel_url(outputs["overlay"]),
        "heatmap_url": rel_url(outputs["heatmap"]),
        "gradcam_url": rel_url(outputs.get("gradcam")) if outputs.get("gradcam") else "",
        "gradcam_overlay_url": rel_url(outputs.get("gradcam_overlay")) if outputs.get("gradcam_overlay") else "",
        "overlay_label": overlay_label,
        "heatmap_label": heatmap_label,
        "xai_method": analysis.get("xai_method", xai_method),
        "xai_comparison": analysis.get("xai_comparison", []),
        "report_url": rel_url(report_path) if report_path.exists() else "",
        "json_url": rel_url(json_path) if json_path.exists() else "",
        "retained_image": retained_image_token(image_path),
        "retained_image_name": Path(image_path).name,
        "form_values": form_values,
    }


def split_defect_summary(text):
    text = text.strip()
    for sep in (",", " - ", ";"):
        if sep in text:
            head, tail = text.split(sep, 1)
            return head.strip(), sep + tail
    return text, ""


def render_llm_report(text):
    rows = []
    known_labels = {"KẾT QUẢ", "VỊ TRÍ", "DẤU HIỆU", "Ý NGHĨA", "KHUYẾN NGHỊ", "LƯU Ý"}
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            label = label.strip().upper()
            value = value.strip()
            if label in known_labels:
                if label == "KẾT QUẢ":
                    defect, rest = split_defect_summary(value)
                    value_html = f"<strong><u>{html.escape(defect)}</u></strong>{html.escape(rest)}"
                else:
                    value_html = html.escape(value)
                rows.append(
                    "<div class='llm-row'>"
                    f"<div class='llm-label'>{html.escape(label)}</div>"
                    f"<div class='llm-value'>{value_html}</div>"
                    "</div>"
                )
                continue
        rows.append(f"<div class='llm-free'>{html.escape(line)}</div>")
    return "<div class='llm-report'><div class='llm-title'>🤖 Câu trả lời từ LLM</div>" + "".join(rows) + "</div>"


def render_report_html(result):
    if result.get("report"):
        if result.get("report_kind") == "llm":
            return render_llm_report(result["report"])
        return "<div class='report'>" + html.escape(result["report"]).replace("\n", "<br>") + "</div>"
    if result.get("message"):
        return "<div class='report warn'>" + html.escape(result["message"]) + "</div>"
    return ""


def format_pct(value):
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def render_xai_comparison_html(result):
    rows = result.get("xai_comparison") or []
    if not rows:
        return ""
    body = []
    for row in rows:
        badges = " ".join(
            f"<span>{html.escape(text)}</span>"
            for text in (row.get("speed_rank"), row.get("reliability_rank"))
            if text
        )
        body.append(
            "<tr>"
            f"<td>{html.escape(row.get('method', ''))}<div class='xai-badges'>{badges}</div></td>"
            f"<td>{float(row.get('runtime_sec') or 0):.2f}s</td>"
            f"<td>{format_pct(row.get('pred_bbox_ratio'))}</td>"
            f"<td>{format_pct(row.get('gt_bbox_ratio'))}</td>"
            f"<td>{format_pct(row.get('reliability_score'))}</td>"
            f"<td>{html.escape(row.get('reliability_label', ''))}</td>"
            "</tr>"
        )
    return (
        "<div class='xai-compare'>"
        "<div class='xai-compare-title'>So sánh XAI</div>"
        "<table><thead><tr>"
        "<th>Phương pháp</th><th>Thời gian</th><th>Tập trung bbox</th>"
        "<th>Trùng GT</th><th>Độ tin cậy</th><th>Mức</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def page(result=None):
    result = result or {}
    report_html = render_report_html(result)
    xai_comparison_html = render_xai_comparison_html(result)
    form_values = result.get("form_values") or {}
    conf_value = html.escape(str(form_values.get("conf", "0.05")), quote=True)
    patch_value = html.escape(str(form_values.get("patch", "120")), quote=True)
    stride_value = html.escape(str(form_values.get("stride", "80")), quote=True)
    max_targets_value = html.escape(str(form_values.get("max_targets", "5")), quote=True)
    explain_value = str(form_values.get("explain_mode", "all"))
    xai_method_value = str(form_values.get("xai_method", "both"))
    explain_all_selected = " selected" if explain_value == "all" else ""
    explain_top_selected = " selected" if explain_value == "top" else ""
    xai_both_selected = " selected" if xai_method_value == "both" else ""
    xai_occ_selected = " selected" if xai_method_value == "occlusion" else ""
    xai_gc_selected = " selected" if xai_method_value == "gradcam" else ""
    retained_image = html.escape(result.get("retained_image", ""), quote=True)
    retained_preview_url = html.escape(result.get("original_url", ""), quote=True) if retained_image else ""
    input_media_title = "Ảnh đang dùng" if retained_image else "Camera trực tiếp"
    retained_preview_html = ""
    if retained_preview_url:
        retained_preview_html = (
            f"""<img id="retainedPreview" class="retained-preview" """
            f"""src="{retained_preview_url}" alt="Ảnh đang dùng">"""
        )

    image_html = ""
    if result.get("ok"):
        overlay_label = result.get("overlay_label", "Vùng được đánh dấu")
        heatmap_label = result.get("heatmap_label", "Bản đồ nổi bật")
        gradcam_html = ""
        if result.get("gradcam_overlay_url"):
            gradcam_html += (
                f"""<figure class="zoomable"><button class="image-zoom" type="button" """
                f"""data-full="{result['gradcam_overlay_url']}" data-title="Grad-CAM overlay">"""
                f"""<img src="{result['gradcam_overlay_url']}" alt="Grad-CAM overlay"></button>"""
                f"""<figcaption>Grad-CAM overlay</figcaption></figure>"""
            )
        if result.get("gradcam_url"):
            gradcam_html += (
                f"""<figure class="zoomable"><button class="image-zoom" type="button" """
                f"""data-full="{result['gradcam_url']}" data-title="Grad-CAM">"""
                f"""<img src="{result['gradcam_url']}" alt="Grad-CAM"></button>"""
                f"""<figcaption>Grad-CAM</figcaption></figure>"""
            )
        image_html = f"""
        <section class="grid">
          <figure class="zoomable"><button class="image-zoom" type="button" data-full="{result['original_url']}" data-title="Ảnh ban đầu"><img src="{result['original_url']}" alt="Ảnh ban đầu"></button><figcaption>Ảnh ban đầu</figcaption></figure>
          <figure class="zoomable"><button class="image-zoom" type="button" data-full="{result['pred_url']}" data-title="Ảnh dự đoán"><img src="{result['pred_url']}" alt="Ảnh dự đoán"></button><figcaption>Ảnh dự đoán</figcaption></figure>
          <figure class="zoomable"><button class="image-zoom" type="button" data-full="{result['overlay_url']}" data-title="{html.escape(overlay_label)}"><img src="{result['overlay_url']}" alt="{html.escape(overlay_label)}"></button><figcaption>{html.escape(overlay_label)}</figcaption></figure>
          <figure class="zoomable"><button class="image-zoom" type="button" data-full="{result['heatmap_url']}" data-title="{html.escape(heatmap_label)}"><img src="{result['heatmap_url']}" alt="{html.escape(heatmap_label)}"></button><figcaption>{html.escape(heatmap_label)}</figcaption></figure>
          {gradcam_html}
        </section>
        <div class="downloads">
          <a href="{result['report_url']}">Tải report</a>
          <a href="{result['json_url']}">Tải JSON</a>
        </div>
        """
    elif result.get("original_url"):
        image_html = f"""
        <section class="grid single">
          <figure class="zoomable"><button class="image-zoom" type="button" data-full="{result['original_url']}" data-title="Ảnh đã tải lên"><img src="{result['original_url']}" alt="Ảnh đã tải lên"></button><figcaption>Ảnh đã tải lên</figcaption></figure>
        </section>
        """

    title = result.get("title", "Tải ảnh hoặc dùng camera để kiểm tra")
    default_url = llm_setting("llm_base_url") or "http://localhost:20128"
    default_key = llm_setting("llm_api_key")
    default_model = llm_setting("llm_model")
    llm_checked = " checked" if LLM_SETTINGS.get("use_llm") else ""
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabric XAI — Industrial Fabric Defect Inspector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ink: #e2e8f0;
      --ink-strong: #f1f5f9;
      --muted: #94a3b8;
      --line: rgba(148,163,184,.18);
      --panel: rgba(15,23,42,.72);
      --accent: #2dd4bf;
      --accent-dim: rgba(45,212,191,.12);
      --accent-2: #fbbf24;
      --surface: #0a0f1a;
      --card-bg: rgba(30,41,59,.55);
      --danger: #fb923c;
      --ok: #34d399;
      --radius: 12px;
      --transition: .25s cubic-bezier(.4,0,.2,1);
    }}
    @keyframes fadeInUp {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }}
    @keyframes pulse-glow {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(45,212,191,.25); }} 50% {{ box-shadow: 0 0 20px 4px rgba(45,212,191,.15); }} }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ height: 100%; }}
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
      background: var(--surface);
      color: var(--ink);
      min-height: 100vh;
      line-height: 1.6;
    }}
    body::before {{
      content: '';
      position: fixed; inset: 0; z-index: -1;
      background:
        radial-gradient(ellipse 60% 40% at 15% 10%, rgba(45,212,191,.08), transparent),
        radial-gradient(ellipse 50% 35% at 85% 5%, rgba(251,191,36,.06), transparent),
        radial-gradient(ellipse 40% 50% at 50% 90%, rgba(99,102,241,.05), transparent);
    }}
    /* ── HEADER ── */
    header {{
      position: relative;
      padding: 28px 36px;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #134e4a 100%);
      border-bottom: 1px solid rgba(45,212,191,.15);
      overflow: hidden;
    }}
    header::after {{
      content: '';
      position: absolute; top: 0; right: 0; bottom: 0; width: 40%;
      background: linear-gradient(135deg, transparent 40%, rgba(45,212,191,.06));
      pointer-events: none;
    }}
    .header-inner {{ display: flex; align-items: center; gap: 16px; position: relative; z-index: 1; }}
    .header-icon {{
      width: 48px; height: 48px;
      background: linear-gradient(135deg, var(--accent), #0d9488);
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; color: #0f172a; font-weight: 900;
      box-shadow: 0 4px 16px rgba(45,212,191,.3);
      flex-shrink: 0;
    }}
    header h1 {{ font-size: 26px; font-weight: 800; color: #f1f5f9; letter-spacing: -.02em; }}
    header h1 span {{ color: var(--accent); }}
    header p {{ margin: 4px 0 0; color: #94a3b8; font-size: 14px; font-weight: 400; }}
    .header-badge {{
      margin-left: auto;
      padding: 6px 14px;
      background: rgba(45,212,191,.1);
      border: 1px solid rgba(45,212,191,.2);
      border-radius: 20px;
      font-size: 12px; font-weight: 600;
      color: var(--accent);
      letter-spacing: .03em;
    }}
    /* ── MAIN ── */
    main {{ padding: 28px 32px 48px; max-width: 1480px; margin: 0 auto; animation: fadeInUp .5s ease-out; }}
    /* ── PANEL ── */
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
      backdrop-filter: blur(16px) saturate(1.2);
      box-shadow: 0 8px 32px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.04);
    }}
    .panel-title {{
      font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
      color: var(--accent); margin-bottom: 18px;
      display: flex; align-items: center; gap: 8px;
    }}
    .panel-title::before {{ content: ''; width: 3px; height: 16px; background: var(--accent); border-radius: 2px; }}
    /* ── FORM ── */
    .input-layout {{ display: grid; grid-template-columns: 1.15fr .85fr; gap: 24px; }}
    .formgrid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; align-items: end; }}
    .form-group {{ margin-bottom: 0; }}
    label {{
      display: block; font-weight: 600; font-size: 13px; margin-bottom: 6px;
      color: var(--muted); letter-spacing: .02em;
    }}
    input[type=file] {{
      width: 100%; padding: 12px 14px;
      border: 1px dashed rgba(148,163,184,.3);
      border-radius: var(--radius);
      background: rgba(30,41,59,.5);
      color: var(--ink); font-family: inherit; font-size: 14px;
      cursor: pointer; transition: border-color var(--transition);
    }}
    input[type=file]:hover {{ border-color: var(--accent); }}
    input[type=file]::file-selector-button {{
      padding: 6px 16px; margin-right: 12px;
      border: none; border-radius: 6px;
      background: linear-gradient(135deg, var(--accent), #0d9488);
      color: #0f172a; font-weight: 600; font-size: 13px; font-family: inherit;
      cursor: pointer;
    }}
    input[type=text], input[type=password], input[type=number], select {{
      width: 100%; padding: 11px 14px;
      border: 1px solid rgba(148,163,184,.2);
      border-radius: 8px;
      background: rgba(30,41,59,.6);
      color: var(--ink-strong);
      font-family: inherit; font-size: 14px;
      transition: border-color var(--transition), box-shadow var(--transition);
    }}
    input[type=text]:focus, input[type=password]:focus, input[type=number]:focus, select:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }}
    select {{ cursor: pointer; }}
    /* ── LLM SECTION ── */
    .llm-section {{ margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--line); }}
    .llm {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 12px; }}
    .llm-toggle {{
      display: flex; align-items: center; gap: 10px;
      font-size: 14px; font-weight: 600; color: var(--ink); cursor: pointer;
    }}
    .llm-toggle input[type=checkbox] {{
      appearance: none; width: 40px; height: 22px;
      background: rgba(148,163,184,.25); border-radius: 11px;
      position: relative; cursor: pointer;
      transition: background var(--transition);
      flex-shrink: 0;
    }}
    .llm-toggle input[type=checkbox]::after {{
      content: ''; position: absolute; top: 3px; left: 3px;
      width: 16px; height: 16px; background: white;
      border-radius: 50%; transition: transform var(--transition);
    }}
    .llm-toggle input[type=checkbox]:checked {{ background: var(--accent); }}
    .llm-toggle input[type=checkbox]:checked::after {{ transform: translateX(18px); }}
    /* ── BUTTONS ── */
    button {{
      border: 0; border-radius: 8px;
      background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 100%);
      color: #0f172a; padding: 12px 22px;
      font-family: inherit; font-weight: 700; font-size: 14px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(45,212,191,.25);
      transition: transform var(--transition), box-shadow var(--transition);
    }}
    button:hover {{ transform: translateY(-1px); box-shadow: 0 6px 20px rgba(45,212,191,.35); }}
    button:active {{ transform: translateY(0); }}
    button.secondary {{
      background: rgba(51,65,85,.7); color: var(--ink);
      box-shadow: none; border: 1px solid var(--line);
    }}
    button.secondary:hover {{ background: rgba(71,85,105,.7); }}
    button.capture {{
      background: linear-gradient(135deg, var(--accent-2), #d97706);
      color: #1c1917;
      box-shadow: 0 4px 14px rgba(251,191,36,.25);
    }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 18px; }}
    /* ── CAMERA ── */
    .camera-box {{
      min-height: 300px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      background: #0f172a;
      position: relative;
    }}
    video, canvas {{ display: block; width: 100%; max-height: 400px; object-fit: contain; background: #0f172a; }}
    canvas {{ display: none; }}
    .retained-preview {{
      display: block;
      width: 100%;
      max-height: 400px;
      min-height: 300px;
      object-fit: contain;
      background: #0f172a;
    }}
    .camera-status {{
      color: var(--muted); font-size: 13px; margin-top: 8px;
      padding: 6px 12px;
      background: rgba(30,41,59,.4);
      border-radius: 6px;
      display: inline-block;
    }}
    /* ── RESULTS GRID ── */
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; animation: fadeInUp .4s ease-out; }}
    .grid.single {{ grid-template-columns: 1fr; max-width: 520px; }}
    figure {{
      margin: 0;
      background: var(--card-bg);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0,0,0,.15);
      transition: transform var(--transition), box-shadow var(--transition);
    }}
    figure:hover {{ transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,.25); }}
    .image-zoom {{
      display: block;
      width: 100%;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
      cursor: zoom-in;
    }}
    .image-zoom:hover {{ transform: none; box-shadow: none; }}
    .image-zoom:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
    figure img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #0f172a; }}
    figcaption {{
      padding: 12px 14px;
      font-weight: 600; font-size: 13px;
      color: var(--ink);
      border-top: 1px solid var(--line);
      text-align: center;
    }}
    /* ── REPORT ── */
    h2 {{ margin-top: 0; color: var(--ink-strong); font-weight: 700; font-size: 20px; }}
    .report {{ line-height: 1.65; font-size: 15px; color: var(--ink); }}
    .xai-compare {{ margin-top: 18px; overflow-x: auto; }}
    .xai-compare-title {{ font-weight: 800; color: var(--ink-strong); margin-bottom: 10px; }}
    .xai-compare table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .xai-compare th, .xai-compare td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      color: var(--ink);
      white-space: nowrap;
    }}
    .xai-compare th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .xai-badges {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 5px; }}
    .xai-badges span {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid rgba(45,212,191,.25);
      background: rgba(45,212,191,.08);
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
    }}
    .llm-report {{ display: grid; gap: 10px; line-height: 1.65; font-size: 15px; }}
    .llm-title {{
      font-size: 18px; font-weight: 800; color: var(--accent);
      margin-bottom: 6px;
      display: flex; align-items: center; gap: 8px;
    }}
    .llm-row {{
      display: grid; grid-template-columns: 140px 1fr; gap: 12px; align-items: start;
      padding: 12px 16px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(30,41,59,.5);
      transition: background var(--transition);
    }}
    .llm-row:hover {{ background: rgba(45,212,191,.06); }}
    .llm-label {{ font-weight: 700; color: var(--accent); font-size: 12px; letter-spacing: .05em; text-transform: uppercase; padding-top: 2px; }}
    .llm-value {{ color: var(--ink); }}
    .llm-value strong {{ font-weight: 800; color: var(--ink-strong); }}
    .llm-value u {{ text-underline-offset: 3px; text-decoration-color: var(--accent); text-decoration-thickness: 2px; }}
    .llm-free {{ padding: 12px 16px; border-radius: 10px; background: rgba(30,41,59,.4); color: var(--ink); }}
    .warn {{ color: var(--danger); }}
    .downloads {{ margin-top: 18px; display: flex; gap: 12px; }}
    .downloads a {{
      padding: 10px 18px;
      background: rgba(45,212,191,.1);
      border: 1px solid rgba(45,212,191,.2);
      border-radius: 8px;
      color: var(--accent);
      text-decoration: none; font-weight: 600; font-size: 14px;
      transition: background var(--transition);
    }}
    .downloads a:hover {{ background: rgba(45,212,191,.18); }}
    .hint {{ color: var(--muted); font-size: 13px; font-style: italic; }}
    .lightbox {{
      position: fixed;
      inset: 0;
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 28px;
      background: rgba(2,6,23,.88);
      backdrop-filter: blur(8px);
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox-inner {{
      width: min(96vw, 1400px);
      max-height: 94vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
    }}
    .lightbox-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      color: white;
      font-weight: 800;
      letter-spacing: .01em;
    }}
    .lightbox-close {{
      width: 40px;
      height: 40px;
      padding: 0;
      border-radius: 999px;
      background: rgba(15,23,42,.82);
      border: 1px solid rgba(148,163,184,.35);
      color: white;
      box-shadow: none;
      font-size: 24px;
      line-height: 1;
      cursor: pointer;
    }}
    .lightbox img {{
      width: 100%;
      max-height: calc(94vh - 58px);
      object-fit: contain;
      border-radius: 10px;
      background: #020617;
      border: 1px solid rgba(148,163,184,.28);
      box-shadow: 0 22px 70px rgba(0,0,0,.42);
    }}
    /* ── RESPONSIVE ── */
    @media (max-width: 1100px) {{ .input-layout, .grid {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 760px) {{
      .input-layout, .formgrid, .llm, .grid, .llm-row {{ grid-template-columns: 1fr; }}
      main {{ padding: 16px; }}
      header {{ padding: 20px; }}
      .header-badge {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div class="header-icon">F</div>
      <div>
        <h1>Fabric <span>XAI</span></h1>
        <p>Hệ thống kiểm tra lỗi vải tự động &mdash; Phân tích XAI &amp; Nhận xét AI cho công nghiệp dệt may</p>
      </div>
      <div class="header-badge">INDUSTRIAL QC</div>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="panel-title">Đầu vào kiểm tra</div>
      <form id="checkForm" method="post" enctype="multipart/form-data" autocomplete="off">
        <div class="input-layout">
          <div>
            <label>Ảnh đầu vào</label>
            <input id="imageInput" name="image" type="file" accept="image/*">
            <div class="formgrid" style="margin-top:14px">
              <div>
                <label>Ngưỡng phát hiện</label>
                <input name="conf" type="number" min="0.01" max="0.80" step="0.01" value="{conf_value}">
              </div>
              <div>
                <label>Vùng che (patch)</label>
                <input name="patch" type="number" min="40" max="240" step="20" value="{patch_value}">
              </div>
              <div>
                <label>Bước quét (stride)</label>
                <input name="stride" type="number" min="40" max="200" step="20" value="{stride_value}">
              </div>
            </div>
            <div class="formgrid" style="margin-top:14px">
              <div>
                <label>Chế độ giải thích</label>
                <select id="explainMode" name="explain_mode">
                  <option value="all"{explain_all_selected}>Tất cả vùng lỗi (chi tiết)</option>
                  <option value="top"{explain_top_selected}>Chỉ vùng chính (nhanh)</option>
                </select>
              </div>
              <div>
                <label>Số vùng tối đa</label>
                <input id="maxTargets" name="max_targets" type="number" min="1" max="10" step="1" value="{max_targets_value}">
              </div>
              <div>
                <label>Phương pháp XAI</label>
                <select id="xaiMethod" name="xai_method">
                  <option value="both"{xai_both_selected}>Cả 2: so sánh</option>
                  <option value="occlusion"{xai_occ_selected}>Chỉ Occlusion</option>
                  <option value="gradcam"{xai_gc_selected}>Chỉ Grad-CAM</option>
                </select>
              </div>
            </div>
            <input id="retainedImage" name="retained_image" type="hidden" value="{retained_image}">
            <input name="xai_scope" type="hidden" value="bbox">
            <input name="roi_scale" type="hidden" value="2.5">
            <input name="min_roi" type="hidden" value="240">
          </div>
          <div>
            <label id="inputMediaTitle">{html.escape(input_media_title)}</label>
            <div class="camera-box">
              {retained_preview_html}
              <video id="camera" autoplay playsinline muted></video>
              <canvas id="snapshot"></canvas>
            </div>
            <input id="cameraImage" name="camera_image" type="hidden">
            <div class="actions">
              <button class="secondary" type="button" id="startCamera">▶ Bật camera</button>
              <button class="capture" type="button" id="captureCamera">&#x1f4f7; Chụp khung hình</button>
            </div>
            <div id="cameraStatus" class="camera-status">Có thể tải ảnh hoặc chụp trực tiếp từ camera.</div>
          </div>
        </div>
        <div class="llm-section">
          <label class="llm-toggle"><input id="useLlm" name="use_llm" type="checkbox"{llm_checked}> Gọi LLM qua 9Router</label>
          <div class="llm">
            <div>
              <label>Base URL</label>
              <input id="llmBaseUrl" name="llm_base_url" type="text" value="{html.escape(default_url)}">
            </div>
            <div>
              <label>API Key</label>
              <input id="llmApiKey" name="llm_api_key" type="password" value="{html.escape(default_key)}">
            </div>
            <div>
              <label>Vision Model ID</label>
              <input id="llmModel" name="llm_model" type="text" value="{html.escape(default_model)}">
            </div>
          </div>
        </div>
        <div class="actions">
          <button type="submit">&#x1f50d; Chạy kiểm tra</button>
          <button class="secondary" type="button" id="clearLlmSettings">Xóa cấu hình LLM</button>
          <span class="hint">Nếu không có kết quả, thử giảm ngưỡng xuống 0.05.</span>
        </div>
      </form>
    </section>
    <section class="panel">
      <div class="panel-title">Kết quả phân tích</div>
      <h2>{html.escape(title)}</h2>{report_html}{xai_comparison_html}
    </section>
    {image_html}
  </main>
  <div id="lightbox" class="lightbox" aria-hidden="true">
    <div class="lightbox-inner" role="dialog" aria-modal="true" aria-labelledby="lightboxTitle">
      <div class="lightbox-bar">
        <div id="lightboxTitle">Ảnh kết quả</div>
        <button id="lightboxClose" class="lightbox-close" type="button" aria-label="Đóng ảnh">×</button>
      </div>
      <img id="lightboxImage" src="" alt="Ảnh xem chi tiết">
    </div>
  </div>
  <script>
    const video = document.getElementById('camera');
    const canvas = document.getElementById('snapshot');
    const hidden = document.getElementById('cameraImage');
    const imageInput = document.getElementById('imageInput');
    const retainedImage = document.getElementById('retainedImage');
    const retainedPreview = document.getElementById('retainedPreview');
    const inputMediaTitle = document.getElementById('inputMediaTitle');
    const statusEl = document.getElementById('cameraStatus');
    const form = document.getElementById('checkForm');
    const explainMode = document.getElementById('explainMode');
    const maxTargets = document.getElementById('maxTargets');
    const useLlm = document.getElementById('useLlm');
    const llmBaseUrl = document.getElementById('llmBaseUrl');
    const llmApiKey = document.getElementById('llmApiKey');
    const llmModel = document.getElementById('llmModel');
    const clearLlmSettings = document.getElementById('clearLlmSettings');
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxTitle = document.getElementById('lightboxTitle');
    const lightboxClose = document.getElementById('lightboxClose');
    const LLM_STORAGE = {{
      useLlm: 'fabric_xai_use_llm',
      baseUrl: 'fabric_xai_llm_base_url',
      apiKey: 'fabric_xai_llm_api_key',
      model: 'fabric_xai_llm_model'
    }};
    const MAX_IMAGE_EDGE = 1600;
    const JPEG_QUALITY = 0.82;
    const COMPRESS_FILE_OVER_BYTES = 8 * 1024 * 1024;
    let stream = null;
    let compressedSubmit = false;

    function scaledSize(width, height) {{
      const edge = Math.max(width, height);
      const ratio = edge > MAX_IMAGE_EDGE ? MAX_IMAGE_EDGE / edge : 1;
      return {{
        width: Math.max(1, Math.round(width * ratio)),
        height: Math.max(1, Math.round(height * ratio))
      }};
    }}

    function drawToCanvas(source, width, height) {{
      const size = scaledSize(width, height);
      canvas.width = size.width;
      canvas.height = size.height;
      canvas.getContext('2d').drawImage(source, 0, 0, size.width, size.height);
      return size;
    }}

    function imageFileToDataUrl(file) {{
      return new Promise((resolve, reject) => {{
        const img = new Image();
        const objectUrl = URL.createObjectURL(file);
        img.onload = () => {{
          try {{
            const size = drawToCanvas(img, img.naturalWidth, img.naturalHeight);
            URL.revokeObjectURL(objectUrl);
            resolve({{ dataUrl: canvas.toDataURL('image/jpeg', JPEG_QUALITY), size }});
          }} catch (err) {{
            URL.revokeObjectURL(objectUrl);
            reject(err);
          }}
        }};
        img.onerror = () => {{
          URL.revokeObjectURL(objectUrl);
          reject(new Error('Cannot read image'));
        }};
        img.src = objectUrl;
      }});
    }}

    function loadLlmSettings() {{
      const savedUse = localStorage.getItem(LLM_STORAGE.useLlm);
      const savedUrl = localStorage.getItem(LLM_STORAGE.baseUrl);
      const savedKey = localStorage.getItem(LLM_STORAGE.apiKey);
      const savedModel = localStorage.getItem(LLM_STORAGE.model);
      if (savedUse !== null) useLlm.checked = savedUse === '1';
      if (savedUrl) llmBaseUrl.value = savedUrl;
      if (savedKey) llmApiKey.value = savedKey;
      if (savedModel) llmModel.value = savedModel;
    }}

    function saveLlmSettings() {{
      localStorage.setItem(LLM_STORAGE.useLlm, useLlm.checked ? '1' : '0');
      if (llmBaseUrl.value.trim()) localStorage.setItem(LLM_STORAGE.baseUrl, llmBaseUrl.value.trim());
      if (llmApiKey.value.trim()) localStorage.setItem(LLM_STORAGE.apiKey, llmApiKey.value.trim());
      if (llmModel.value.trim()) localStorage.setItem(LLM_STORAGE.model, llmModel.value.trim());
    }}

    async function clearSavedLlmSettings() {{
      Object.values(LLM_STORAGE).forEach(key => localStorage.removeItem(key));
      useLlm.checked = false;
      llmApiKey.value = '';
      llmModel.value = '';
      try {{
        await fetch('/clear-llm-settings', {{ method: 'POST' }});
      }} catch (err) {{}}
      statusEl.textContent = 'Đã xóa cấu hình LLM lưu trên trình duyệt.';
    }}

    loadLlmSettings();
    [useLlm, llmBaseUrl, llmApiKey, llmModel].forEach(el => {{
      el.addEventListener('change', saveLlmSettings);
      el.addEventListener('input', saveLlmSettings);
    }});
    clearLlmSettings.addEventListener('click', clearSavedLlmSettings);

    function openLightbox(src, title) {{
      lightboxImage.src = src;
      lightboxTitle.textContent = title || 'Ảnh kết quả';
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }}

    function closeLightbox() {{
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      lightboxImage.src = '';
      document.body.style.overflow = '';
    }}

    document.querySelectorAll('.image-zoom').forEach(button => {{
      button.addEventListener('click', () => {{
        openLightbox(button.dataset.full, button.dataset.title);
      }});
    }});

    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (event) => {{
      if (event.target === lightbox) closeLightbox();
    }});
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox();
    }});

    document.getElementById('startCamera').addEventListener('click', async () => {{
      try {{
        if (stream) stream.getTracks().forEach(track => track.stop());
        stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'environment' }}, audio: false }});
        video.srcObject = stream;
        video.style.display = 'block';
        canvas.style.display = 'none';
        if (retainedPreview) retainedPreview.style.display = 'none';
        inputMediaTitle.textContent = 'Camera trực tiếp';
        statusEl.textContent = 'Camera đang bật.';
      }} catch (err) {{
        statusEl.textContent = 'Không mở được camera. Hãy kiểm tra quyền truy cập camera.';
      }}
    }});

    document.getElementById('captureCamera').addEventListener('click', () => {{
      if (!video.videoWidth) {{
        statusEl.textContent = 'Camera chưa sẵn sàng.';
        return;
      }}
      const size = drawToCanvas(video, video.videoWidth, video.videoHeight);
      hidden.value = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
      retainedImage.value = '';
      canvas.style.display = 'block';
      video.style.display = 'none';
      if (retainedPreview) retainedPreview.style.display = 'none';
      inputMediaTitle.textContent = 'Ảnh vừa chụp';
      imageInput.value = '';
      statusEl.textContent = 'Đã chụp và nén ảnh (' + size.width + 'x' + size.height + '). Bấm Chạy kiểm tra.';
    }});

    imageInput.addEventListener('change', () => {{
      hidden.value = '';
      retainedImage.value = '';
      canvas.style.display = 'none';
      if (retainedPreview) retainedPreview.style.display = 'none';
      if (!imageInput.files.length) return;
      if (imageInput.files[0].size > COMPRESS_FILE_OVER_BYTES) {{
        inputMediaTitle.textContent = 'Ảnh tải lên';
        statusEl.textContent = 'Ảnh lớn sẽ được nén tự động trước khi gửi.';
      }} else {{
        inputMediaTitle.textContent = 'Ảnh tải lên';
        statusEl.textContent = 'Đang dùng ảnh tải lên.';
      }}
    }});

    explainMode.addEventListener('change', () => {{
      if (explainMode.value === 'top') {{
        maxTargets.value = '1';
      }} else if (Number(maxTargets.value || 0) < 5) {{
        maxTargets.value = '5';
      }}
    }});

    form.addEventListener('submit', async (event) => {{
      saveLlmSettings();
      if (compressedSubmit) return;
      const file = imageInput.files[0];
      if (!file || file.size <= COMPRESS_FILE_OVER_BYTES) return;
      event.preventDefault();
      statusEl.textContent = 'Ảnh lớn đang được nén trước khi gửi...';
      try {{
        const compressed = await imageFileToDataUrl(file);
        hidden.value = compressed.dataUrl;
        imageInput.value = '';
        canvas.style.display = 'block';
        video.style.display = 'none';
        if (retainedPreview) retainedPreview.style.display = 'none';
        inputMediaTitle.textContent = 'Ảnh tải lên đã nén';
        statusEl.textContent = 'Đã nén ảnh còn ' + compressed.size.width + 'x' + compressed.size.height + '. Đang chạy kiểm tra...';
        compressedSubmit = true;
        if (form.requestSubmit) form.requestSubmit();
        else form.submit();
      }} catch (err) {{
        statusEl.textContent = 'Không nén được ảnh. Đang gửi ảnh gốc...';
        compressedSubmit = true;
        form.submit();
      }}
    }});
  </script>
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return page()

    save_llm_settings(request.form)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    camera_path = save_camera_image(request.form.get("camera_image"))
    if camera_path:
        file_path = camera_path
    else:
        upload = request.files.get("image")
        if upload and upload.filename:
            file_path = UPLOAD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_file_name(upload.filename)}"
            upload.save(file_path)
        else:
            file_path = retained_image_path(request.form.get("retained_image"))
            if not file_path:
                return page({"ok": False, "message": "Chưa có ảnh đầu vào."})
    with PROCESS_LOCK:
        result = run_pipeline(file_path, request.form)
    return page(result)


@app.route("/clear-llm-settings", methods=["POST"])
def clear_llm_settings():
    LLM_SETTINGS.update({
        "use_llm": False,
        "llm_base_url": os.getenv("NINEROUTER_URL", "http://localhost:20128"),
        "llm_api_key": "",
        "llm_model": "",
    })
    return {"ok": True}


@app.route("/files/<path:relpath>")
def serve_file(relpath):
    path = (ROOT / relpath).resolve()
    allowed = DEFAULT_OUT.resolve()
    if allowed not in path.parents and path != allowed:
        return "Not allowed", 403
    if not path.exists():
        return "Not found", 404
    return send_file(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    app.run(host=args.server_name, port=args.server_port, debug=False)


if __name__ == "__main__":
    main()
