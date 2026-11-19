"""
Benchmark XAI tren tap test.
Chay Occlusion + Grad-CAM -> tinh metric -> xuat bang tong hop.

Cach chay:
    # From the repository root
    python Code\\xai\\benchmark_xai.py --max-images 10 --progress

Ket qua luu tai:
    Code\\xai\\xai_results\\benchmark\\benchmark_results.json
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO

from xai_metrics import (
    deletion_auc,
    faithfulness_test,
    gradcam_heatmap,
    insertion_auc,
    pointing_game,
)
from yolo_xai_llm import (
    IMG_EXTS,
    detection_evaluation_for_target,
    detection_summary,
    heat_ratio,
    image_to_label_path,
    load_names,
    no_prediction_evaluation,
    occlusion_heatmap,
    predict_boxes,
    read_yolo_labels,
    target_score,
)


ROOT = Path(__file__).resolve().parents[2]
XAI_DIR = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def output_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "code":
        return ROOT / path
    return XAI_DIR / path


def mean_or_none(values):
    return round(float(np.mean(values)), 3) if values else None


def main():
    p = argparse.ArgumentParser(description="Benchmark XAI tren tap test")
    p.add_argument("--weights", default="Code/xai/model/best.pt")
    p.add_argument("--data", default="voc2007_yolo/dataset.yaml")
    p.add_argument("--source", default="voc2007_yolo/images/test")
    p.add_argument("--out", default=str(XAI_DIR / "xai_results" / "benchmark"))
    p.add_argument("--max-images", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--patch", type=int, default=120)
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--eval-metrics", action="store_true",
                   help="Tinh Deletion/Insertion AUC va Faithfulness (cham hon)")
    p.add_argument("--faithfulness-threshold", type=float, default=0.15,
                   help="Nguong drop confidence de dat faithfulness, vi du 0.15 = giam 15%%")
    p.add_argument("--eval-iou", type=float, default=0.50,
                   help="Nguong IoU de tinh TP/FP/FN/TN")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    weights = project_path(args.weights)
    data_yaml = project_path(args.data)
    source = project_path(args.source)
    out_dir = output_path(args.out)

    model = YOLO(str(weights))
    names = load_names(data_yaml)

    images = sorted(
        img for img in source.rglob("*") if img.suffix.lower() in IMG_EXTS
    )
    images = images[: args.max_images]

    if not images:
        print("Khong tim thay anh nao trong", source)
        return 1

    xai_args = SimpleNamespace(
        imgsz=640,
        conf=args.conf,
        iou=0.45,
        device=None,
        patch=args.patch,
        stride=args.stride,
        xai_scope="full",
        roi_scale=3.0,
        min_roi=240,
        match_iou=0.30,
        eval_iou=args.eval_iou,
        fill_mode="gray",
        eval_metrics=args.eval_metrics,
        faithfulness_threshold=args.faithfulness_threshold,
        progress=args.progress,
    )

    results = []
    for idx, img_path in enumerate(images, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gts = read_yolo_labels(image_to_label_path(img_path), w, h, names)
        preds = predict_boxes(model, image, names, 640, args.conf, 0.45, None)
        if not preds:
            det_eval = no_prediction_evaluation(gts, xai_args)
            print(f"[{idx}/{len(images)}] {img_path.name}: no prediction ({det_eval['case_type']})")
            results.append({
                "image": img_path.name,
                "case_type": det_eval["case_type"],
                "class": "No prediction",
                "conf": 0.0,
                "gt_iou": 0.0,
                "num_ground_truth": len(gts),
                "detection_summary": detection_summary(preds, gts, xai_args),
            })
            continue
        target_det = preds[0]
        gt, gt_iou, det_eval = detection_evaluation_for_target(target_det, gts, xai_args)

        print(
            f"[{idx}/{len(images)}] {img_path.name}: "
            f"{target_det.name} {target_det.conf:.3f}",
            flush=True,
        )

        # Occlusion heatmap
        heat_occ = occlusion_heatmap(
            model, image, names, target_det, xai_args
        )

        # Grad-CAM heatmap
        heat_gc = None
        try:
            heat_gc = gradcam_heatmap(model, image, target_det)
            print("  Grad-CAM: done", flush=True)
        except Exception as e:
            import traceback
            print(f"  Grad-CAM skipped: {e}\n{traceback.format_exc()}")

        row = {
            "image": img_path.name,
            "case_type": det_eval["case_type"],
            "class": target_det.name,
            "conf": round(target_det.conf, 3),
            "detection_summary": detection_summary(preds, gts, xai_args),
            "occ_pred_ratio": round(
                heat_ratio(heat_occ, target_det.xyxy), 3
            ),
        }
        if heat_gc is not None:
            row["gc_pred_ratio"] = round(
                heat_ratio(heat_gc, target_det.xyxy), 3
            )

        if gt:
            row["gt_iou"] = round(gt_iou, 3)
            row["occ_gt_ratio"] = round(
                heat_ratio(heat_occ, gt.xyxy), 3
            )
            row["pg_occ"] = pointing_game(heat_occ, gt.xyxy, w, h)
            if heat_gc is not None:
                row["gc_gt_ratio"] = round(
                    heat_ratio(heat_gc, gt.xyxy), 3
                )
                row["pg_gc"] = pointing_game(heat_gc, gt.xyxy, w, h)

        if args.eval_metrics:
            print("  Deletion AUC ...", flush=True)
            del_auc, _ = deletion_auc(
                model, image, names, target_det, heat_occ, target_score, xai_args
            )
            print("  Insertion AUC ...", flush=True)
            ins_auc, _ = insertion_auc(
                model, image, names, target_det, heat_occ, target_score, xai_args
            )
            faith = faithfulness_test(
                model, image, names, target_det, heat_occ, target_score, xai_args
            )
            row["deletion_auc"] = round(del_auc, 3)
            row["insertion_auc"] = round(ins_auc, 3)
            row["faithful"] = bool(faith["is_faithful"])
            row["conf_drop_pct"] = faith["conf_drop_pct"]

        results.append(row)

    # Tong hop
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print("Khong co ket qua nao.")
        return 1

    # Tinh trung binh
    occ_pred = [r["occ_pred_ratio"] for r in results if "occ_pred_ratio" in r]
    gc_pred = [
        r["gc_pred_ratio"]
        for r in results
        if r.get("gc_pred_ratio") is not None
    ]
    occ_gt = [r["occ_gt_ratio"] for r in results if "occ_gt_ratio" in r]
    gc_gt = [r["gc_gt_ratio"] for r in results if "gc_gt_ratio" in r]
    pg_occ = [r["pg_occ"] for r in results if "pg_occ" in r]
    pg_gc = [r["pg_gc"] for r in results if "pg_gc" in r]
    deletion = [r["deletion_auc"] for r in results if "deletion_auc" in r]
    insertion = [r["insertion_auc"] for r in results if "insertion_auc" in r]
    faithful = [float(r["faithful"]) for r in results if "faithful" in r]
    case_counts = {}
    for row in results:
        case = row.get("case_type", "unknown")
        case_counts[case] = case_counts.get(case, 0) + 1

    summary = {
        "num_images": len(results),
        "num_xai_images": len(occ_pred),
        "case_counts": case_counts,
        "avg_occ_pred_ratio": mean_or_none(occ_pred),
        "avg_gc_pred_ratio": mean_or_none(gc_pred),
        "avg_occ_gt_ratio": mean_or_none(occ_gt),
        "avg_gc_gt_ratio": mean_or_none(gc_gt),
        "pointing_game_occ_accuracy": mean_or_none(pg_occ),
        "pointing_game_gc_accuracy": mean_or_none(pg_gc),
        "avg_deletion_auc": mean_or_none(deletion),
        "avg_insertion_auc": mean_or_none(insertion),
        "faithfulness_rate": mean_or_none(faithful),
        "per_image": results,
    }

    # In bang ket qua
    has_gc = any(r.get("gc_pred_ratio") is not None for r in results)
    header = f"{'Image':<35} {'Case':<5} {'Class':<14} {'Conf':>5}"
    header += f" {'OccP':>6} {'OccG':>6} {'PGo':>4}"
    if has_gc:
        header += f" {'GcP':>6} {'GcG':>6} {'PGg':>4}"
    if args.eval_metrics:
        header += f" {'DelA':>6} {'InsA':>6} {'Fth':>4}"

    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(header)
    print(sep)

    for r in results:
        line = (
            f"{r['image'][:34]:<35} {r.get('case_type', '-'):<5} "
            f"{r['class']:<14} {r['conf']:>5.3f}"
        )
        if "occ_pred_ratio" not in r:
            print(line)
            continue
        line += f" {r['occ_pred_ratio']:>5.1%}"
        line += f" {r.get('occ_gt_ratio', 0):>5.1%}"
        pg_o = r.get("pg_occ", 0)
        line += f" {'V' if pg_o == 1 else 'X':>4}"
        if has_gc:
            gc_p = r.get("gc_pred_ratio")
            gc_g = r.get("gc_gt_ratio")
            pg_g = r.get("pg_gc")
            line += (
                f" {gc_p:>5.1%}" if gc_p is not None else f" {'N/A':>6}"
            )
            line += (
                f" {gc_g:>5.1%}" if gc_g is not None else f" {'N/A':>6}"
            )
            if pg_g is not None:
                line += f" {'V' if pg_g == 1 else 'X':>4}"
            else:
                line += f" {'-':>4}"
        if args.eval_metrics:
            line += f" {r.get('deletion_auc', 0):>6.3f}"
            line += f" {r.get('insertion_auc', 0):>6.3f}"
            line += f" {'V' if r.get('faithful') else 'X':>4}"
        print(line)

    print(sep)
    avg_line = f"{'TRUNG BINH':<35} {'':5} {'':14} {'':>5}"
    avg_line += f" {np.mean(occ_pred) if occ_pred else 0:>5.1%}"
    avg_ogt = np.mean(occ_gt) if occ_gt else 0
    avg_line += f" {avg_ogt:>5.1%}"
    avg_pgo = np.mean(pg_occ) if pg_occ else 0
    avg_line += f" {avg_pgo:>4.0%}"
    if has_gc:
        avg_line += f" {np.mean(gc_pred) if gc_pred else 0:>5.1%}"
        avg_line += f" {np.mean(gc_gt) if gc_gt else 0:>5.1%}"
        avg_line += f" {np.mean(pg_gc) if pg_gc else 0:>4.0%}"
    if args.eval_metrics:
        avg_line += f" {np.mean(deletion) if deletion else 0:>6.3f}"
        avg_line += f" {np.mean(insertion) if insertion else 0:>6.3f}"
        avg_line += f" {np.mean(faithful) if faithful else 0:>4.0%}"
    print(avg_line)
    print(f"{'=' * len(header)}")

    # Luu JSON
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDa luu: {json_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
