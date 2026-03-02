# video-detect-main/webcam_sync_compare.py
import os, sys, time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd
import python_speech_features

from pathlib import Path

# ---- SyncNet import path ----
ROOT = Path(__file__).resolve().parent
SYNCNET_REPO = (ROOT / "syncnet_python").resolve()
sys.path.insert(0, str(SYNCNET_REPO))

from SyncNetInstance import SyncNetInstance, calc_pdist

# ----------------------------
def mfcc_20(audio_f32, sr=16000):
    """audio_f32: float32 mono [-1,1]"""
    # python_speech_features.mfcc expects float or int; we'll pass float
    mfcc = python_speech_features.mfcc(audio_f32, sr).T.astype(np.float32)  # (13, T)
    # SyncNet expects (1,1,13,20) for a 5-frame chunk (20 mfcc frames)
    if mfcc.shape[1] > 20:
        mfcc = mfcc[:, :20]
    else:
        mfcc = np.pad(mfcc, ((0,0), (0, 20-mfcc.shape[1])))
    return mfcc

@torch.inference_mode()
def confidence_from_buffers(model: SyncNetInstance, frames_rgb, audio_f32, device, vshift=10):
    """
    frames_rgb: list of RGB frames (H,W,3), length >= 5
    audio_f32: mono float32 buffer aligned roughly to frames window
    Returns: conf(float), min_dist(float)
    """
    if len(frames_rgb) < 5:
        return 0.0, 10.0

    # 1) video tensor: take first 5 frames only (simple)
    frames_5 = frames_rgb[:5]
    frames_5 = [cv2.resize(f, (224,224)) for f in frames_5]
    v = np.transpose(np.array(frames_5).astype(np.float32), (3,0,1,2))  # (3,5,224,224)
    v = torch.from_numpy(v).unsqueeze(0).to(device)                    # (1,3,5,224,224)

    # 2) audio tensor -> mfcc -> (1,1,13,20)
    mfcc = mfcc_20(audio_f32, sr=16000)
    a = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,13,20)

    # 3) features
    v_feat = F.normalize(model.__S__.forward_lip(v), p=2, dim=1)
    a_feat = F.normalize(model.__S__.forward_aud(a), p=2, dim=1)

    # 4) dist + conf (네가 쓰던 방식 유지)
    dist = float(F.pairwise_distance(v_feat, a_feat).item())
    conf = max(0.0, 10.0 * (1.0 - dist / 2.0))
    return conf, dist

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_model", required=True, help="path to syncnet_v2.model")
    ap.add_argument("--new_model", required=True, help="path to finetuned .pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--win_sec", type=float, default=2.0)
    ap.add_argument("--audio_sr", type=int, default=16000)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- load models ----
    net_old = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024, device=str(device))
    net_old.loadParameters(args.old_model)  # .model은 이걸로

    net_new = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024, device=str(device))
    state = torch.load(args.new_model, map_location="cpu")             # pth is SyncNetInstance state_dict
    net_new.load_state_dict(state, strict=False)

    net_old.to(device)
    net_new.to(device)
    net_old.eval()
    net_new.eval()

    # ---- buffers ----
    max_frames = int(args.fps * args.win_sec)
    frames_buf = deque(maxlen=max_frames)

    max_audio = int(args.audio_sr * args.win_sec)
    audio_buf = deque(maxlen=max_audio)

    def audio_cb(indata, frames, time_info, status):
        if status:
            pass
        # indata: (frames, channels)
        x = indata[:, 0].astype(np.float32)
        audio_buf.extend(x.tolist())

    stream = sd.InputStream(
        channels=1,
        samplerate=args.audio_sr,
        callback=audio_cb,
        blocksize=int(args.audio_sr * 0.05),  # 50ms
    )
    stream.start()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Press 'q' to quit.")
    last_t = time.time()
    fps_ema = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_buf.append(frame_rgb)

            # fps
            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / max(dt, 1e-6)
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst

            # compute when buffers ready
            if len(frames_buf) >= 5 and len(audio_buf) >= int(args.audio_sr * 0.25):
                audio_np = np.array(audio_buf, dtype=np.float32)
                old_conf, old_dist = confidence_from_buffers(net_old, list(frames_buf), audio_np, device)
                new_conf, new_dist = confidence_from_buffers(net_new, list(frames_buf), audio_np, device)
                diff = new_conf - old_conf
            else:
                old_conf = new_conf = diff = 0.0

            # overlay
            out = frame_bgr.copy()
            cv2.putText(out, f"FPS: {fps_ema:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, f"OLD conf: {old_conf:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            cv2.putText(out, f"NEW conf: {new_conf:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            cv2.putText(out, f"DIFF    : {diff:+.3f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,200), 2)

            cv2.imshow("SyncNet Webcam Compare", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        stream.stop()
        stream.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()