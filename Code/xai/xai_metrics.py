"""
Bo sung phuong phap XAI va metric danh gia cho du an kiem tra loi vai.

Phuong phap XAI:
  - Grad-CAM: gradient-based, nhanh, can truy cap backbone YOLO

Metric danh gia:
  - Pointing Game (Zhang et al., ECCV 2016)
  - Deletion AUC (Petsiuk et al., BMVC 2018)
  - Insertion AUC (Petsiuk et al., BMVC 2018)
  - Faithfulness Test
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _trapz(y, x):
    """NumPy 2.x renamed trapz to trapezoid; keep old envs working."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ============================================================
# GRAD-CAM
# ============================================================


def _find_target_layer(torch_model, preferred=9):
    """Tim layer conv cuoi backbone cua YOLO."""
    try:
        layer = torch_model.model[preferred]
        if any(isinstance(m, torch.nn.Conv2d) for m in layer.modules()):
            return layer
    except (IndexError, AttributeError):
        pass
    for i in range(min(15, len(torch_model.model)) - 1, -1, -1):
        m = torch_model.model[i]
        if any(isinstance(sub, torch.nn.Conv2d) for sub in m.modules()):
            return m
    raise RuntimeError("Khong tim duoc layer phu hop cho Grad-CAM")


def _nearest_anchor_index(head_out, target, image_shape, imgsz):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = target.xyxy
    cx = ((x1 + x2) / 2.0) * imgsz / max(w, 1)
    cy = ((y1 + y2) / 2.0) * imgsz / max(h, 1)

    feats = head_out.get("feats") if isinstance(head_out, dict) else None
    if isinstance(feats, (list, tuple)) and feats:
        best_idx = 0
        best_dist = float("inf")
        offset = 0
        for feat in feats:
            if not hasattr(feat, "shape") or len(feat.shape) < 4:
                continue
            _, _, fh, fw = feat.shape
            stride_x = imgsz / max(int(fw), 1)
            stride_y = imgsz / max(int(fh), 1)
            ix = int(np.clip(round(cx / stride_x - 0.5), 0, fw - 1))
            iy = int(np.clip(round(cy / stride_y - 0.5), 0, fh - 1))
            anchor_cx = (ix + 0.5) * stride_x
            anchor_cy = (iy + 0.5) * stride_y
            dist = (anchor_cx - cx) ** 2 + (anchor_cy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = offset + iy * fw + ix
            offset += fh * fw
        return best_idx

    boxes = head_out.get("boxes") if isinstance(head_out, dict) else None
    if boxes is not None and boxes.dim() == 3 and boxes.shape[-1] > 0:
        box_centers = boxes[0, :2, :]
        dists = (box_centers[0] - cx) ** 2 + (box_centers[1] - cy) ** 2
        return int(torch.argmin(dists).item())

    return None


def gradcam_heatmap(model, image, target, layer_index=9, imgsz=640):
    """
    Tao Grad-CAM heatmap cho 1 detection target tren YOLO.

    Nguyen ly:
    1. Forward pass -> lay feature map o layer cuoi backbone
    2. Backward pass tu class score cua target
    3. GAP gradient -> trong so moi channel
    4. Weighted sum + ReLU -> resize ve kich thuoc anh goc

    Args:
        model: ultralytics YOLO model
        image: numpy array BGR (H, W, 3)
        target: DetBox object voi .cls va .conf
        layer_index: index cua layer backbone (mac dinh 9 cho YOLOv8)
        imgsz: kich thuoc input cho model

    Returns:
        heatmap numpy array (H, W), gia tri trong [0, 1]
    """
    torch_model = model.model
    target_layer = _find_target_layer(torch_model, layer_index)

    activations = {}
    gradients_store = {}

    def fwd_hook(m, inp, out):
        tensor = out[0] if isinstance(out, (list, tuple)) else out
        activations["val"] = tensor
        if tensor.requires_grad:
            tensor.register_hook(lambda grad: gradients_store.update({"val": grad}))

    h1 = target_layer.register_forward_hook(fwd_hook)

    was_training = torch_model.training
    original_requires_grad = {}
    original_head_training = None
    original_head_end2end = None
    original_cv2 = None
    original_cv3 = None
    head = None
    if hasattr(torch_model, "model") and hasattr(torch_model.model[-1], "training"):
        head = torch_model.model[-1]
        original_head_training = head.training
        original_head_end2end = getattr(head, "end2end", None)
        original_cv2 = getattr(head, "cv2", None)
        original_cv3 = getattr(head, "cv3", None)

    try:
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = torch.nn.functional.interpolate(
            img_t, size=(imgsz, imgsz), mode="bilinear", align_corners=False
        )
        device = next(torch_model.parameters()).device
        img_t = img_t.to(device)
        img_t.requires_grad_(True)

        torch_model.eval()
        if head is not None:
            head.training = True
            if (
                getattr(head, "end2end", False)
                and (getattr(head, "cv2", None) is None or getattr(head, "cv3", None) is None)
            ):
                head.cv2 = getattr(head, "one2one_cv2", None)
                head.cv3 = getattr(head, "one2one_cv3", None)
                head.end2end = False
        torch_model.zero_grad(set_to_none=True)

        for name, p in torch_model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad_(True)

        with torch.enable_grad():
            preds = torch_model(img_t)

            cls_idx = target.cls
            if isinstance(preds, dict):
                # YOLOv10 / v11 with training=True returns a dict
                head_out = preds
                if "one2many" in preds and "scores" in preds["one2many"]:
                    head_out = preds["one2many"]
                elif "one2one" in preds and "scores" in preds["one2one"]:
                    head_out = preds["one2one"]

                if isinstance(head_out, dict) and "scores" in head_out:
                    # scores shape: [batch, nc, N]
                    cls_scores = head_out["scores"][0, cls_idx, :]
                    anchor_idx = _nearest_anchor_index(head_out, target, image.shape, imgsz)
                else:
                    cls_scores = preds[0].flatten() if isinstance(preds, (list, tuple)) else preds.flatten()
                    anchor_idx = None
            else:
                pred = preds[0] if isinstance(preds, (list, tuple)) else preds
                # Xu ly output shape: [batch, 4+nc, N] hoac [batch, N, 4+nc] hoac [batch, 300, 6] (YOLOv10)
                if pred.shape[-1] == 6:
                    # YOLOv10: [x1, y1, x2, y2, conf, cls]
                    mask = (pred[..., 5] == cls_idx)
                    cls_scores = (pred[..., 4] * mask.float()).flatten()
                elif pred.dim() == 3:
                    if pred.shape[1] > pred.shape[2]:
                        cls_scores = pred[0, :, 4 + cls_idx]
                    else:
                        cls_scores = pred[0, 4 + cls_idx, :]
                else:
                    cls_scores = pred.flatten()
                anchor_idx = None

            if cls_scores.numel() == 0:
                raise RuntimeError("Grad-CAM khong co class score de backward")
            if anchor_idx is None:
                anchor_idx = int(torch.argmax(cls_scores).item())
            anchor_idx = int(max(0, min(cls_scores.numel() - 1, anchor_idx)))
            score = cls_scores[anchor_idx]
            score.backward()

        if "val" not in activations or "val" not in gradients_store:
            raise RuntimeError("Grad-CAM khong thu duoc activation/gradient")
        act = activations["val"].detach()
        grad = gradients_store["val"].detach()

        # GAP gradient -> trong so alpha cho moi channel
        weights = grad.mean(dim=(2, 3), keepdim=True)

        # Weighted combination + ReLU
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        return cam.astype(np.float32)

    finally:
        torch_model.train(was_training)
        if original_head_training is not None:
            torch_model.model[-1].training = original_head_training
        if original_head_end2end is not None:
            torch_model.model[-1].end2end = original_head_end2end
            torch_model.model[-1].cv2 = original_cv2
            torch_model.model[-1].cv3 = original_cv3
        for name, p in torch_model.named_parameters():
            if name in original_requires_grad:
                p.requires_grad_(original_requires_grad[name])
        h1.remove()


def gradcam_multi(model, image, targets):
    """Grad-CAM cho nhieu targets, combine bang max."""
    h, w = image.shape[:2]
    cams = []
    for t in targets:
        try:
            cams.append(gradcam_heatmap(model, image, t))
        except Exception:
            pass
    if not cams:
        return np.zeros((h, w), dtype=np.float32)
    combined = np.maximum.reduce(cams)
    if combined.max() > 0:
        combined /= combined.max()
    return combined


# ============================================================
# METRIC DANH GIA XAI DINH LUONG
# ============================================================


def pointing_game(heat, gt_xyxy, w, h):
    """
    Pointing Game: pixel co gia tri cao nhat co nam trong GT bbox?

    Returns: 1.0 (hit) hoac 0.0 (miss)

    Tham khao: Zhang et al.,
    "Top-Down Neural Attention by Excitation Backprop", ECCV 2016
    """
    if heat.max() <= 0:
        return 0.0
    max_y, max_x = np.unravel_index(np.argmax(heat), heat.shape)
    x1 = int(max(0, min(w - 1, round(gt_xyxy[0]))))
    y1 = int(max(0, min(h - 1, round(gt_xyxy[1]))))
    x2 = int(max(0, min(w - 1, round(gt_xyxy[2]))))
    y2 = int(max(0, min(h - 1, round(gt_xyxy[3]))))
    return 1.0 if (x1 <= max_x <= x2 and y1 <= max_y <= y2) else 0.0


def deletion_auc(model, image, names, target, heat, score_fn, args, steps=50):
    """
    Deletion: xoa dan pixel quan trong nhat -> confidence phai giam.

    AUC thap = heatmap tot (xoa dung vung quan trong -> confidence giam nhanh).

    score_fn: target_score function tu yolo_xai_llm

    Tham khao: Petsiuk et al.,
    "RISE: Randomized Input Sampling for Explanation of Black-box Models",
    BMVC 2018
    """
    h, w = image.shape[:2]
    n_pixels = h * w
    order = np.argsort(-heat.flatten())
    scores = []
    step_size = max(1, n_pixels // steps)

    for i in range(0, n_pixels + 1, step_size):
        modified = image.copy()
        if i > 0:
            indices = order[: min(i, n_pixels)]
            ys, xs = np.unravel_index(indices, (h, w))
            modified[ys, xs] = [127, 127, 127]
        sc = score_fn(model, modified, names, target, args)
        scores.append(sc)

    base = scores[0] if scores[0] > 0 else 1e-8
    normed = [s / base for s in scores]
    x_axis = np.linspace(0, 1, len(normed))
    return float(_trapz(normed, x_axis)), normed


def insertion_auc(model, image, names, target, heat, score_fn, args, steps=50):
    """
    Insertion: them dan pixel quan trong nhat -> confidence phai tang.

    AUC cao = heatmap tot (them dung vung quan trong -> confidence tang nhanh).

    Tham khao: Petsiuk et al., "RISE", BMVC 2018
    """
    h, w = image.shape[:2]
    n_pixels = h * w
    baseline = np.full_like(image, 127)
    order = np.argsort(-heat.flatten())
    scores = []
    step_size = max(1, n_pixels // steps)

    for i in range(0, n_pixels + 1, step_size):
        modified = baseline.copy()
        if i > 0:
            indices = order[: min(i, n_pixels)]
            ys, xs = np.unravel_index(indices, (h, w))
            modified[ys, xs] = image[ys, xs]
        sc = score_fn(model, modified, names, target, args)
        scores.append(sc)

    base = score_fn(model, image, names, target, args)
    base = base if base > 0 else 1e-8
    normed = [s / base for s in scores]
    x_axis = np.linspace(0, 1, len(normed))
    return float(_trapz(normed, x_axis)), normed


def faithfulness_test(model, image, names, target, heat, score_fn, args,
                      threshold=0.5):
    """
    Faithfulness test: xoa vung heatmap > threshold -> confidence phai giam.

    is_faithful = True neu confidence giam lon hon nguong cau hinh so voi ban dau.
    """
    h, w = image.shape[:2]
    threshold_val = np.percentile(heat[heat > 0], 75) if (heat > 0).any() else threshold
    important_mask = heat >= threshold_val
    area = float(important_mask.sum()) / (h * w)

    masked = image.copy()
    masked[important_mask] = [127, 127, 127]

    original_conf = target.conf
    masked_conf = score_fn(model, masked, names, target, args)
    drop = original_conf - masked_conf
    faithfulness_threshold = float(getattr(args, "faithfulness_threshold", 0.15))

    return {
        "original_conf": round(original_conf, 4),
        "masked_conf": round(masked_conf, 4),
        "conf_drop": round(drop, 4),
        "conf_drop_pct": f"{drop / max(original_conf, 1e-8):.1%}",
        "faithfulness_threshold": round(faithfulness_threshold, 4),
        "is_faithful": drop > faithfulness_threshold * original_conf,
        "important_area_ratio": round(area, 4),
    }


# ============================================================
# REPORT SO SANH PHUONG PHAP XAI
# ============================================================


def comparison_report(analysis):
    """Tao bao cao so sanh Occlusion vs Grad-CAM + metric danh gia."""
    occ = analysis.get("occlusion_heat_in_pred_bbox_ratio")
    gc = analysis.get("gradcam_heat_in_pred_bbox_ratio")
    metrics = analysis.get("xai_metrics", {})

    if occ is None and gc is None and not metrics:
        return ""

    lines = ["", "--- So sanh phuong phap XAI ---"]

    if occ is not None and gc is not None:
        lines.append(f"Occlusion Sensitivity: {occ:.1%} nang luong trong bbox")
        lines.append(f"Grad-CAM:              {gc:.1%} nang luong trong bbox")
        diff = abs(occ - gc)
        if diff < 0.15:
            lines.append(
                "Nhan xet: hai phuong phap cho ket qua nhat quan."
            )
        else:
            lines.append(
                "Nhan xet: hai phuong phap khac biet, "
                "nen xem ky overlay cua ca hai."
            )
    elif occ is not None:
        lines.append(f"Occlusion Sensitivity: {occ:.1%} nang luong trong bbox")
    elif gc is not None:
        lines.append(f"Grad-CAM: {gc:.1%} nang luong trong bbox")

    if "pointing_game_occlusion" in metrics:
        hit = "Hit" if metrics["pointing_game_occlusion"] == 1.0 else "Miss"
        lines.append(f"Pointing Game (Occlusion): {hit}")
    if "pointing_game_gradcam" in metrics:
        hit = "Hit" if metrics["pointing_game_gradcam"] == 1.0 else "Miss"
        lines.append(f"Pointing Game (Grad-CAM):  {hit}")
    if "deletion_auc" in metrics:
        lines.append(
            f"Deletion AUC:  {metrics['deletion_auc']:.3f} (thap = tot)"
        )
    if "insertion_auc" in metrics:
        lines.append(
            f"Insertion AUC: {metrics['insertion_auc']:.3f} (cao = tot)"
        )
    faith = metrics.get("faithfulness")
    if faith:
        status = "DAT" if faith["is_faithful"] else "CHUA DAT"
        lines.append(
            f"Faithfulness:  conf {faith['original_conf']:.3f} -> "
            f"{faith['masked_conf']:.3f} "
            f"(giam {faith['conf_drop_pct']}, {status})"
        )

    return "\n".join(lines)
