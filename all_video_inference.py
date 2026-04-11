import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from collections import OrderedDict

from model.autoencoder import *
from model.video_swin_transformer import *
from model.utils import np_load_frame, compute_motion_mask
from utils import psnr, anomaly_score_list, AUC
from torchvision import transforms

# -------------------------------
# ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser(description="Batch Video Inference")

# python batch_video_inference.py \
#     --test_folder "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video_128\ipad_half_video\R02\testing\frames" \
#     --label_folder "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video_128\ipad_half_video\R02\test_label" \
#     --model_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\downloaded_models\model_final.pth" \
#     --psnr_threshold 30.0 \
#     --mem_dim 2000 \
#     --num_frames 8

parser.add_argument('--test_folder', type=str, required=True,
                    help='path to testing/frames containing video folders (01, 02, ...)')
parser.add_argument('--label_folder', type=str, required=True,
                    help='path to test_label containing .npy files')
parser.add_argument('--model_path', type=str, required=True)


parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--mem_dim', type=int, default=2000)

parser.add_argument('--motion_mask', action='store_true')
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--mask_ratio', type=float, default=0.5)

parser.add_argument('--use_skip', action='store_true')
parser.add_argument('--use_wavelet', action='store_true')

parser.add_argument('--save_dir', type=str, default='batch_inference_results',
                    help='directory to save plots and results')

args = parser.parse_args()
# transform = transforms.ToTensor()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Setup output directory
# -------------------------------
model_filename = os.path.splitext(os.path.basename(args.model_path))[0]
save_dir = os.path.join(args.save_dir, model_filename)
os.makedirs(save_dir, exist_ok=True)


threshold_path = f"threshold_info_{model_filename}.npy"
threshold_data = np.load(threshold_path, allow_pickle=True).item()
psnr_threshold = threshold_data["psnr_threshold"]

print("-"*100)
print(f"USING PSNR THRESHOLD {psnr_threshold} db")
print("-"*100)


# -------------------------------
# Load model
# -------------------------------
model = VST(
    mem_dim=args.mem_dim,
    use_wavelet=args.use_wavelet,
    use_skip=args.use_skip
)

# model_dict = torch.load(args.model_path, map_location=DEVICE, weights_only=False)

# try:
#     model.load_state_dict(model_dict['model'].state_dict(), strict=False)
# except:
#     model.load_state_dict(model_dict['model'], strict=False)

# model = model.to(DEVICE)
# model.eval()


ckpt = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
state = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']

# Strip DataParallel's "module." prefix if present
new_state = OrderedDict()
for k, v in state.items():
    new_state[k[7:] if k.startswith('module.') else k] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)
print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
assert len(missing) == 0 and len(unexpected) == 0, \
    f"Checkpoint load mismatch!\nmissing: {missing[:5]}\nunexpected: {unexpected[:5]}"

model = model.to(DEVICE).eval()

loss_func = nn.MSELoss(reduction='none')

# counter = 1
# -------------------------------
# Helper functions
# -------------------------------
def load_clip(frame_paths, start_idx):
    batch = []
    raw_frames = []

    for i in range(start_idx, start_idx + args.num_frames):
        img = np_load_frame(frame_paths[i], args.h, args.w)

        if args.motion_mask:
            raw_frames.append(img)

        # img = torch.from_numpy(img).permute(2, 0, 1)
        # img = transform(img) 
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        batch.append(img)
        # if counter==1:
        # print('\n----------check image range----------------------------------------------\n')
        # print("RANGE:", img.min().item(), img.max().item())
        # print('\n----------check image range----------------------------------------------\n')
            # counter+=1

    batch = torch.stack(batch, dim=1)

    if args.motion_mask:
        mid_idx = args.num_frames // 2
        mask = compute_motion_mask(
            raw_frames, mid_idx,
            args.block_size, args.mask_ratio
        )
        mask = torch.from_numpy(mask).to(DEVICE)
        mask = mask.unsqueeze(0).unsqueeze(0)
        batch = batch * mask

    return batch.unsqueeze(0)


def compute_psnr(mse_val):
    return 10 * np.log10(1.0 / (mse_val + 1e-8))


def run_single_video(video_folder, label_path):
    """Run inference on a single video folder, return psnr list and gt list."""
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    if len(frame_paths) == 0:
        print(f"  No frames found in {video_folder}, skipping.")
        return None, None

    gt_labels = np.load(label_path)

    # Sync lengths
    min_len = min(len(frame_paths), len(gt_labels))
    frame_paths = frame_paths[:min_len]
    gt_labels = gt_labels[:min_len]

    psnr_values = []
    gt_list = []
    middle_offset = args.num_frames // 2

    with torch.no_grad():
        for i in range(len(frame_paths) - args.num_frames + 1):
            clip = load_clip(frame_paths, i).to(DEVICE)

            outputs = model(clip)
            recon = outputs['output']

            mid_idx = args.num_frames // 2
            mse = loss_func(recon[0, :, mid_idx], clip[0, :, mid_idx])
            recon_loss = torch.mean(mse).item()

            psnr_val = compute_psnr(recon_loss)
            psnr_values.append(psnr_val)

            label_idx = min(i + middle_offset, len(gt_labels) - 1)
            gt_list.append(gt_labels[label_idx])

    return psnr_values, gt_list


def plot_video_result(video_name, psnr_values, gt_list, threshold):
    """Save normalized anomaly score plot and raw PSNR plot for one video."""
    psnr_arr = np.array(psnr_values)
    gt_arr = np.array(gt_list)
    predictions = psnr_arr < threshold

    # Find anomaly GT segments for pink highlighting
    rect_start, rect_end = [], []
    active = False
    for i, val in enumerate(gt_arr):
        if val == 1 and not active:
            rect_start.append(i)
            active = True
        elif val == 0 and active:
            rect_end.append(i)
            active = False
    if active:
        rect_end.append(len(gt_arr) - 1)

    # --- Raw PSNR plot with threshold ---
    plt.figure(figsize=(12, 5))
    plt.plot(psnr_arr, label='PSNR', color='green')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')

    ax = plt.gca()
    y_min, y_max = psnr_arr.min(), psnr_arr.max()
    margin = (y_max - y_min) * 0.05
    for rs, re in zip(rect_start, rect_end):
        ax.add_patch(Rectangle((rs, y_min - margin), re - rs,
                                y_max - y_min + 2 * margin,
                                facecolor='pink', alpha=0.4))

    # Prediction overlay
    for i, pred in enumerate(predictions):
        if pred:
            plt.axvspan(i, i + 1, color='orange', alpha=0.15)

    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.title(f'Video: {video_name} — GT (pink) vs Prediction (orange)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"single_video_{video_name}.png"), dpi=150)
    plt.close()


# -------------------------------
# Discover all video folders
# -------------------------------
video_folders = sorted([
    d for d in glob.glob(os.path.join(args.test_folder, '*'))
    if os.path.isdir(d)
])

print(f"Found {len(video_folders)} video folders in {args.test_folder}")
print(f"PSNR Threshold: {psnr_threshold:.4f}")
print(f"Results will be saved to: {save_dir}\n")

# -------------------------------
# Run inference on all videos
# -------------------------------
all_psnrs = []
all_gt = []
video_results = OrderedDict()

for vf in video_folders:
    video_name = os.path.basename(vf)
    label_file = f"{int(video_name):03d}.npy"
    label_path = os.path.join(args.label_folder, label_file)

    if not os.path.exists(label_path):
        print(f"[SKIP] Label not found for {video_name}: {label_path}")
        continue

    print(f"[{video_name}] Running inference...")
    psnr_values, gt_list = run_single_video(vf, label_path)

    if psnr_values is None:
        continue

    video_results[video_name] = {
        'psnr': psnr_values,
        'gt': gt_list
    }

    # Per-video plot
    plot_video_result(video_name, psnr_values, gt_list, psnr_threshold)
    print(f"  Frames: {len(psnr_values)}, Plot saved.")

    all_psnrs.extend(psnr_values)
    all_gt.extend(gt_list)

# -------------------------------
# Global AUC
# -------------------------------
# if len(all_psnrs) > 0:
#     anomaly_scores = anomaly_score_list(all_psnrs)
#     auc = AUC(anomaly_scores, np.expand_dims(1 - np.array(all_gt), 0))
#     print(f"\nGlobal AUC across all videos: {auc * 100:.2f}%")

#     # Per-video AUC
#     print("\nPer-video AUC:")
#     for vname, vdata in video_results.items():
#         v_scores = anomaly_score_list(vdata['psnr'])
#         v_auc = AUC(v_scores, np.expand_dims(1 - np.array(vdata['gt']), 0))
#         print(f"  {vname}: {v_auc * 100:.2f}%")
# else:
#     print("No results to compute AUC.")

print(f"\nAll plots saved to {save_dir}/")