import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import argparse

from model.autoencoder import *
from model.video_swin_transformer import *
from utils import np_load_frame, compute_motion_mask  # 🔥 IMPORTANT

# -------------------------------
# 🔧 ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--video_folder', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--model_filename', type=str, required=True)

parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--mem_dim', type=int, default=2000)

# Motion mask args (same as training)
parser.add_argument('--motion_mask', action='store_true')
parser.add_argument('--block_size', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.5)

# VST args (CRITICAL)
parser.add_argument('--use_skip', action='store_true')
parser.add_argument('--use_wavelet', action='store_true')

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔹 Load threshold
# -------------------------------
threshold_path = f"threshold_info_{args.model_filename}.npy"
threshold_data = np.load(threshold_path, allow_pickle=True).item()
PSNR_THRESHOLD = threshold_data["psnr_threshold"]

print(f"Loaded PSNR Threshold: {PSNR_THRESHOLD:.4f}")

# -------------------------------
# 🔹 Load model (MATCH TRAINING)
# -------------------------------
model = VST(
    mem_dim=args.mem_dim,
    use_wavelet=args.use_wavelet,
    use_skip=args.use_skip
)

model_dict = torch.load(args.model_path, map_location=DEVICE, weights_only=False)

try:
    model.load_state_dict(model_dict['model'].state_dict(), strict=False)
except:
    model.load_state_dict(model_dict['model'], strict=False)

model = model.to(DEVICE)
model.eval()

loss_func = nn.MSELoss(reduction='none')

# -------------------------------
# 🔹 Load frames (TRAINING STYLE)
# -------------------------------
frame_paths = sorted(glob.glob(os.path.join(args.video_folder, "*.jpg")))

def load_clip(start_idx):
    batch = []
    raw_frames = []

    for i in range(start_idx, start_idx + args.num_frames):
        img = np_load_frame(frame_paths[i], args.h, args.w)  # 🔥 SAME AS TRAIN

        if args.motion_mask:
            raw_frames.append(img)

        img = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,C) → (C,H,W)
        batch.append(img)

    batch = torch.stack(batch, dim=1)  # (C, T, H, W)

    if args.motion_mask:
        mid_idx = args.num_frames // 2
        mask = compute_motion_mask(
            raw_frames,
            mid_idx,
            args.block_size,
            args.mask_ratio
        )
        mask = torch.from_numpy(mask).to(DEVICE)

        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        batch = batch * mask  # apply mask like training

    return batch.unsqueeze(0)  # (1,C,T,H,W)

# -------------------------------
# 🔹 PSNR
# -------------------------------
def psnr(mse):
    return 10 * np.log10(1.0 / (mse + 1e-8))

# -------------------------------
# 🔹 Inference
# -------------------------------
psnr_values = []
predictions = []

with torch.no_grad():
    for i in range(len(frame_paths) - args.num_frames + 1):
        clip = load_clip(i).to(DEVICE)

        outputs = model(clip)
        recon = outputs['output']

        mid_idx = args.num_frames // 2

        mse = loss_func(
            recon[0, :, mid_idx],
            clip[0, :, mid_idx]
        )

        recon_loss = torch.mean(mse).item()
        psnr_val = psnr(recon_loss)

        psnr_values.append(psnr_val)

        is_anomaly = psnr_val < PSNR_THRESHOLD
        predictions.append(is_anomaly)

        print(f"Frame {i:04d} | PSNR: {psnr_val:.2f} | {'ANOMALY' if is_anomaly else 'NORMAL'}")

# -------------------------------
# 🔹 Plot
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(psnr_values, label="PSNR", color="green")
plt.axhline(y=PSNR_THRESHOLD, color="red", linestyle="--", label="Threshold")

plt.xlabel("Frame")
plt.ylabel("PSNR")
plt.title("Single Video Anomaly Detection")

for i, val in enumerate(predictions):
    if val:
        plt.axvspan(i, i+1, color='red', alpha=0.2)

plt.legend()
plt.savefig("single_video_result.png")
plt.show()

print("\n✅ Done. Plot saved as single_video_result.png")