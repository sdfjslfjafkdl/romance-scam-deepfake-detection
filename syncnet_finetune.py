"""SyncNet fine-tuning (PyTorch)

This file is a cleaned-up Python script version of the notebook `SyncNet_FineTuning.ipynb`.

What it includes
- A Dataset that reads (video.mp4, audio.wav) pairs from:
    <root_dir>/syncnet_data/positive/<sample_id>/{video.mp4,audio.wav}
    <root_dir>/syncnet_data/negative/<sample_id>/<sub_id>/{video.mp4,audio.wav}
- A simple contrastive fine-tuning loop using in-batch shuffling for negatives.

What it intentionally omits
- Colab/Drive specific setup commands and dataset "organizing" utilities
- One-off evaluation / confidence-score testing cells

You can adapt paths and hyperparameters from the CLI examples at the bottom.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import soundfile as sf
import python_speech_features

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Import SyncNetInstance (supports both "flat" and package-style layouts)
# -----------------------------------------------------------------------------
try:
    # If your repo has: syncnet_python/SyncNetInstance.py
    from syncnet_python.SyncNetInstance import SyncNetInstance  # type: ignore
except Exception:
    # If SyncNetInstance.py is on PYTHONPATH
    from SyncNetInstance import SyncNetInstance  # type: ignore


class SyncNetDataset(Dataset):
    """Loads SyncNet pairs from a folder structure (positive/negative)."""

    def __init__(self, root_dir: str, video_fps: int = 25, audio_sr: int = 16000):
        self.root_dir = root_dir
        self.video_fps = video_fps
        self.audio_sr = audio_sr
        self.samples: List[Dict[str, str | int]] = []

        target_root = os.path.join(root_dir, "syncnet_data")
        pos_root = os.path.join(target_root, "positive")
        neg_root = os.path.join(target_root, "negative")

        def is_valid(name: str) -> bool:
            # Ignore macOS metadata files like "._*"
            return not name.startswith("._")

        # Positive samples
        if os.path.isdir(pos_root):
            for sid in sorted(os.listdir(pos_root)):
                if not is_valid(sid):
                    continue
                p_path = os.path.join(pos_root, sid)
                v_p = os.path.join(p_path, "video.mp4")
                a_p = os.path.join(p_path, "audio.wav")
                if os.path.exists(v_p) and os.path.exists(a_p):
                    self.samples.append({"video": v_p, "audio": a_p, "label": 1})

        # Negative samples (nested)
        if os.path.isdir(neg_root):
            for sid in sorted(os.listdir(neg_root)):
                if not is_valid(sid):
                    continue
                n_path = os.path.join(neg_root, sid)
                if not os.path.isdir(n_path):
                    continue
                for sub in os.listdir(n_path):
                    if not is_valid(sub):
                        continue
                    sub_path = os.path.join(n_path, sub)
                    v_n = os.path.join(sub_path, "video.mp4")
                    a_n = os.path.join(sub_path, "audio.wav")
                    if os.path.exists(v_n) and os.path.exists(a_n):
                        self.samples.append({"video": v_n, "audio": a_n, "label": -1})

        print(f"✅ Dataset ready: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_5frames(self, video_path: str) -> torch.Tensor:
        """Load 5 frames and return tensor (C, T, H, W) with T=5, H=W=224."""
        cap = cv2.VideoCapture(video_path)
        frames: List[np.ndarray] = []
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        # Pad if insufficient frames
        while len(frames) < 5:
            if len(frames) == 0:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                frames.append(frames[-1].copy())

        v = np.array(frames).astype(np.float32)  # (T, H, W, C)
        try:
            v = np.transpose(v, (3, 0, 1, 2))  # (C, T, H, W)
            return torch.from_numpy(v)
        except Exception as e:
            print(f"[WARN] video load failed: {video_path} ({e})")
            return torch.zeros((3, 5, 224, 224), dtype=torch.float32)

    def _load_audio_mfcc(self, wav_path: str) -> torch.Tensor:
        """Load wav and return MFCC tensor shaped (1, C, T)."""
        audio, _ = sf.read(wav_path)
        mfcc = python_speech_features.mfcc(audio, self.audio_sr).T.astype(np.float32)  # (C, T)

        # Match notebook logic: keep/pad to T=20
        if mfcc.shape[1] > 20:
            mfcc = mfcc[:, :20]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, 20 - mfcc.shape[1])))

        a = torch.from_numpy(mfcc).unsqueeze(0)  # (1, C, T)
        return a

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        v = self._load_video_5frames(str(s["video"]))
        a = self._load_audio_mfcc(str(s["audio"]))
        label = torch.tensor(float(s["label"]), dtype=torch.float32)
        return v, a, label


@dataclass
class FinetuneConfig:
    base_model_path: str
    save_path: str
    root_dir: str
    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-5
    margin: float = 1.0
    num_workers: int = 2


def finetune_syncnet(cfg: FinetuneConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / Loader
    dataset = SyncNetDataset(root_dir=cfg.root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Model
    net = SyncNetInstance(dropout=0, num_layers_in_fc_layers=1024).to(device)
    net.loadParameters(cfg.base_model_path)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    net.train()

    print("🚀 Fine-tuning start (in-batch shuffling negatives)")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        steps = 0

        for v, a, label in dataloader:
            v = v.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Extract & normalize features
            v_feat = F.normalize(net.__S__.forward_lip(v), p=2, dim=1)
            a_feat = F.normalize(net.__S__.forward_aud(a), p=2, dim=1)

            # Positive: only where label == 1
            dist_pos = F.pairwise_distance(v_feat, a_feat)
            if (label == 1).any():
                loss_pos = torch.pow(dist_pos[label == 1], 2).mean()
            else:
                loss_pos = torch.tensor(0.0, device=device)

            # Negative: roll audio features within batch (in-batch shuffling)
            a_feat_neg = torch.roll(a_feat, shifts=1, dims=0)
            dist_neg = F.pairwise_distance(v_feat, a_feat_neg)
            loss_neg = torch.pow(torch.clamp(cfg.margin - dist_neg, min=0.0), 2).mean()

            loss = loss_pos + loss_neg
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            steps += 1

        if steps > 0 and ((epoch + 1) % 2 == 0 or epoch == 0 or epoch + 1 == cfg.epochs):
            print(f"Epoch [{epoch+1:02d}/{cfg.epochs}]  avg_loss={epoch_loss/steps:.6f}")

    # Save weights
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
    torch.save(net.state_dict(), cfg.save_path)
    print(f"✅ Saved finetuned weights -> {cfg.save_path}")


if __name__ == "__main__":
    # Example (edit paths for your machine/repo):
    # python syncnet_finetune.py \
    #   --root_dir /path/to/syncnet_dataset \
    #   --base_model_path ./syncnet_python/data/syncnet_v2.model \
    #   --save_path ./syncnet_python/data/syncnet_finetuned.pth
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True, help="Dataset root dir that contains syncnet_data/")
    p.add_argument("--base_model_path", required=True, help="Path to syncnet_v2.model (pretrained)")
    p.add_argument("--save_path", required=True, help="Where to save finetuned .pth")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    finetune_syncnet(
        FinetuneConfig(
            base_model_path=args.base_model_path,
            save_path=args.save_path,
            root_dir=args.root_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            num_workers=args.num_workers,
        )
    )