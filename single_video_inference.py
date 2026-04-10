
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
from model.utils import np_load_frame, compute_motion_mask

# -------------------------------
# 🔧 ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()

# python single_video_inference.py --video_folder "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video_128\ipad_half_video\R02\testing\frames\04" --label_folder "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video_128\ipad_half_video\R02\test_label" --model_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\downloaded_models\model_final R2_waveletLoss.pth" --mem_dim 2000 --num_frames 8

parser.add_argument('--video_folder', type=str, required=True)
parser.add_argument('--label_folder', type=str, required=True)  
parser.add_argument('--model_path', type=str, required=True)


parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--mem_dim', type=int, default=2000)

parser.add_argument('--motion_mask', action='store_true')
parser.add_argument('--block_size', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.5)

parser.add_argument('--use_skip', action='store_true')
parser.add_argument('--use_wavelet', action='store_true')

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔹 Load threshold
# -------------------------------
model_filename = os.path.splitext(os.path.basename(args.model_path))[0]
threshold_path = f"threshold_info_{model_filename}.npy"
threshold_data = np.load(threshold_path, allow_pickle=True).item()
psnr_threshold = threshold_data["psnr_threshold"]

print(f"Loaded PSNR Threshold: {psnr_threshold:.4f}")

# -------------------------------
# 🔹 Load model
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
# 🔹 Load frames
# -------------------------------
frame_paths = sorted(glob.glob(os.path.join(args.video_folder, "*.jpg")))

# -------------------------------
# 🔹 Load labels (same as loader)
# -------------------------------
video_name = os.path.basename(args.video_folder)
label_file = f"{int(video_name):03d}.npy"
label_path = os.path.join(args.label_folder, label_file)

if not os.path.exists(label_path):
    raise ValueError(f"Label file not found: {label_path}")

gt_labels = np.load(label_path)

# CRITICAL FIX: sync lengths
min_len = min(len(frame_paths), len(gt_labels))
frame_paths = frame_paths[:min_len]
gt_labels = gt_labels[:min_len]

print(f"Synced frames & labels length: {min_len}")

# -------------------------------
# 🔹 Load clip
# -------------------------------
def load_clip(start_idx):
    batch = []
    raw_frames = []

    for i in range(start_idx, start_idx + args.num_frames):
        img = np_load_frame(frame_paths[i], args.h, args.w)

        if args.motion_mask:
            raw_frames.append(img)

        img = torch.from_numpy(img).permute(2, 0, 1)
        batch.append(img)

    batch = torch.stack(batch, dim=1)

    if args.motion_mask:
        mid_idx = args.num_frames // 2
        mask = compute_motion_mask(
            raw_frames,
            mid_idx,
            args.block_size,
            args.mask_ratio
        )
        mask = torch.from_numpy(mask).to(DEVICE)
        mask = mask.unsqueeze(0).unsqueeze(0)
        batch = batch * mask

    return batch.unsqueeze(0)

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
gt_list = []

middle_offset = args.num_frames // 2

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

        # 🔥 Prediction
        is_anomaly = psnr_val < psnr_threshold
        predictions.append(is_anomaly)

        # 🔥 Ground truth (aligned like loader)
        label_idx = i + middle_offset
        safe_idx = min(label_idx, len(gt_labels) - 1)
        gt = gt_labels[safe_idx]

        gt_list.append(gt)

        print(f"Frame {i:04d} | PSNR: {psnr_val:.2f} | {'ANOMALY' if is_anomaly else 'NORMAL'}")

# -------------------------------
# 🔹 Plot
# -------------------------------
plt.figure(figsize=(12, 5))

plt.plot(psnr_values, label="PSNR", color="green")
plt.axhline(y=psnr_threshold, color="red", linestyle="--", label="Threshold")

#  Ground truth
for i, val in enumerate(gt_list):
    if val == 1:
        plt.axvspan(i, i+1, color='red', alpha=0.25)

#  Predictions
for i, val in enumerate(predictions):
    if val:
        plt.axvspan(i, i+1, color='orange', alpha=0.25)

plt.xlabel("Frame")
plt.ylabel("PSNR")
plt.title("GT (Red) vs Prediction (Orange)")
plt.legend()
plt.savefig(f"single_video_{video_name}.png")
plt.show()

print("\n Done. Plot saved as single_video_result.png")

