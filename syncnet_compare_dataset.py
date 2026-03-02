# video-detect-main/syncnet_compare_dataset.py
import os, sys, csv, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import python_speech_features
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SYNCNET_REPO = (ROOT / "syncnet_python").resolve()
DATA_DIR_DEFAULT = (ROOT / "romance_detection" / "syncnet_data").resolve()

sys.path.insert(0, str(SYNCNET_REPO))
from SyncNetInstance import SyncNetInstance


# -----------------------------
def calculate_confidence(video_path: str, audio_path: str, model: SyncNetInstance, device):
    """
    returns (conf, dist)
    conf: 0~10 (높을수록 sync)
    dist: L2 distance (낮을수록 sync)
    """
    # 1) read 5 frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()

    if len(frames) < 5:
        return 0.0, 10.0

    v = torch.from_numpy(
        np.transpose(np.array(frames).astype(np.float32), (3, 0, 1, 2))
    ).unsqueeze(0).to(device)

    # 2) audio -> mfcc (13, T) -> (1,1,13,20)
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        return 0.0, 10.0
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    mfcc = python_speech_features.mfcc(audio, 16000).T.astype(np.float32)
    if mfcc.shape[1] > 20:
        mfcc = mfcc[:, :20]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 20 - mfcc.shape[1])))

    a = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        v_feat = F.normalize(model.__S__.forward_lip(v), p=2, dim=1)
        a_feat = F.normalize(model.__S__.forward_aud(a), p=2, dim=1)

        dist = float(F.pairwise_distance(v_feat, a_feat).item())
        conf = max(0.0, 10.0 * (1.0 - dist / 2.0))
    return conf, dist


def find_pairs_in_dir(root_dir: Path):
    """
    root_dir 아래에서 video.* + audio.* 가 같이 있는 폴더를 찾는다.
    (예: .../positive/xxx/video.mp4 + audio.wav)
    """
    pairs = []
    root_dir = Path(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        fn = set(filenames)

        vid = None
        for c in ["video.mp4", "video.mov", "video.mkv", "video.avi"]:
            if c in fn:
                vid = c
                break

        aud = None
        for c in ["audio.wav", "audio.flac", "audio.mp3"]:
            if c in fn:
                aud = c
                break

        if vid and aud:
            pairs.append((Path(dirpath) / vid, Path(dirpath) / aud))

    return pairs


def label_from_path(p: Path):
    s = str(p).lower()
    if "positive" in s:
        return "positive"
    if "negative" in s or "unsync" in s or "neg_" in s:
        return "negative"
    return "unknown"


def mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def plot_and_save(rows, out_dir: Path):
    """
    rows: list[dict]
    out_dir: where to save pngs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # split
    pos = [r for r in rows if r["label"] == "positive"]
    neg = [r for r in rows if r["label"] == "negative"]
    unk = [r for r in rows if r["label"] == "unknown"]

    # 1) Histogram of diff (all)
    diffs = [r["diff"] for r in rows]
    plt.figure()
    plt.hist(diffs, bins=30)
    plt.title("SyncNet Confidence Difference (new - old)")
    plt.xlabel("diff")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_diff_all.png", dpi=200)
    plt.close()

    # 2) Boxplot of diff by label
    groups = []
    labels = []
    if pos:
        groups.append([r["diff"] for r in pos])
        labels.append("positive")
    if neg:
        groups.append([r["diff"] for r in neg])
        labels.append("negative")
    if unk:
        groups.append([r["diff"] for r in unk])
        labels.append("unknown")

    if groups:
        plt.figure()
        plt.boxplot(groups, labels=labels, showmeans=True)
        plt.axhline(0.0, linewidth=1)
        plt.title("Diff by Label (new - old)")
        plt.ylabel("diff")
        plt.tight_layout()
        plt.savefig(out_dir / "box_diff_by_label.png", dpi=200)
        plt.close()

    # 3) Scatter old vs new
    x = [r["conf_old"] for r in rows]
    y = [r["conf_new"] for r in rows]
    plt.figure()
    plt.scatter(x, y, s=12)
    plt.plot([0, 10], [0, 10])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Old vs New Confidence")
    plt.xlabel("old")
    plt.ylabel("new")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_old_vs_new.png", dpi=200)
    plt.close()

    # 4) Top-k examples (largest increase/decrease) -> text file
    rows_sorted_inc = sorted(rows, key=lambda r: r["diff"], reverse=True)
    rows_sorted_dec = sorted(rows, key=lambda r: r["diff"])

    def dump_top(rows_sorted, name, k=10):
        lines = []
        for r in rows_sorted[:k]:
            lines.append(
                f"{r['label']}\tdiff={r['diff']:+.4f}\told={r['conf_old']:.4f}\tnew={r['conf_new']:.4f}\tvideo={r['video']}"
            )
        (out_dir / name).write_text("\n".join(lines), encoding="utf-8")

    dump_top(rows_sorted_inc, "top10_increase.txt", k=10)
    dump_top(rows_sorted_dec, "top10_decrease.txt", k=10)


def main():
    import argparse
    ap = argparse.ArgumentParser()

    # ✅ data_root 기본값을 syncnet_python/data로 고정
    ap.add_argument(
        "--data_root",
        default=str(DATA_DIR_DEFAULT),
        help="target folder containing pairs (default: romance_detection/syncnet_data)",
    )

    ap.add_argument("--old_model", required=True, help="path to syncnet_v2.model")
    ap.add_argument("--new_model", required=True, help="path to finetuned .pth")

    ap.add_argument("--out_dir", default="syncnet_compare_outputs", help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load models ----
    net_old = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024, device=str(device))
    net_old.loadParameters(args.old_model)

    net_new = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024, device=str(device))
    state = torch.load(args.new_model, map_location="cpu")
    net_new.load_state_dict(state, strict=False)

    net_old.to(device).eval()
    net_new.to(device).eval()

    # ---- find pairs only within data_root ----
    data_root = Path(args.data_root).resolve()
    pairs = find_pairs_in_dir(data_root)
    if not pairs:
        print(f"No pairs found under: {data_root}")
        print("Expected directories containing video.* and audio.*")
        return

    rows = []
    for vpath, apath in pairs:
        lab = label_from_path(vpath)
        c_old, d_old = calculate_confidence(str(vpath), str(apath), net_old, device)
        c_new, d_new = calculate_confidence(str(vpath), str(apath), net_new, device)

        row = {
            "label": lab,
            "video": str(vpath),
            "audio": str(apath),
            "conf_old": float(c_old),
            "conf_new": float(c_new),
            "diff": float(c_new - c_old),
            "dist_old": float(d_old),
            "dist_new": float(d_new),
        }
        rows.append(row)

        print(f"[{lab}] diff={row['diff']:+.3f} old={row['conf_old']:.3f} new={row['conf_new']:.3f} | {vpath}")

    # ---- save CSV ----
    csv_path = out_dir / "syncnet_compare_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary json/txt ----
    pos = [r for r in rows if r["label"] == "positive"]
    neg = [r for r in rows if r["label"] == "negative"]
    unk = [r for r in rows if r["label"] == "unknown"]

    summary = {
        "data_root": str(data_root),
        "count_total": len(rows),
        "count_positive": len(pos),
        "count_negative": len(neg),
        "count_unknown": len(unk),
        "mean_diff_all": mean(r["diff"] for r in rows),
        "mean_diff_positive": mean(r["diff"] for r in pos) if pos else None,
        "mean_diff_negative": mean(r["diff"] for r in neg) if neg else None,
        "mean_conf_old_all": mean(r["conf_old"] for r in rows),
        "mean_conf_new_all": mean(r["conf_new"] for r in rows),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("==== Summary ====")
    lines.append(f"Data root: {data_root}")
    lines.append(f"Total: {summary['count_total']} | Positive: {summary['count_positive']} | Negative: {summary['count_negative']} | Unknown: {summary['count_unknown']}")
    lines.append(f"Mean diff (all): {summary['mean_diff_all']:+.4f}")
    if summary["mean_diff_positive"] is not None:
        lines.append(f"Mean diff (positive): {summary['mean_diff_positive']:+.4f}   (want +)")
    if summary["mean_diff_negative"] is not None:
        lines.append(f"Mean diff (negative): {summary['mean_diff_negative']:+.4f}   (want -)")
    lines.append(f"Mean conf old/new (all): {summary['mean_conf_old_all']:.4f} -> {summary['mean_conf_new_all']:.4f}")
    lines.append(f"CSV saved: {csv_path}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- plots ----
    plot_and_save(rows, out_dir)

    print("\n".join(lines))
    print(f"\nPlots saved under: {out_dir}")
    print(" - hist_diff_all.png")
    print(" - box_diff_by_label.png")
    print(" - scatter_old_vs_new.png")
    print(" - top10_increase.txt / top10_decrease.txt")


if __name__ == "__main__":
    main()