"""
evaluate_simple.py

Minimal evaluation script for SimpleVST_AE.
Computes AUC using PSNR-based anomaly scores.

Usage:
  python evaluate_simple.py \
    --dataset_path /path/to/R01 \
    --dataset_type VAD \
    --model_dir ./exp/log_simple_vst/model_50.pth \
    --num_frames 16
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from model.utils import TestDataLoader
from model.simple_vst_ae import SimpleVST_AE
from utils import *

import time
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import OrderedDict

# python evaluate_simple.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01"  --model_dir "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\ipad_repo\exp\log_simple_vst\model_02.pth" --num_workers 0

# ─── Args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simple VST AE Evaluation")
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--dataset_type', type=str, default='VAD')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--print_time', action='store_true')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Model ──────────────────────────────────────────────────────
model = SimpleVST_AE()

if torch.cuda.is_available():
    model = nn.DataParallel(model).to(device)

model_dict = torch.load(args.model_dir, weights_only=False)
try:
    model.load_state_dict(model_dict['model'].state_dict())
except:
    model.load_state_dict(model_dict['model'])

model.eval()
loss_func_mse = nn.MSELoss(reduction='none')

# ─── Dataset ─────────────────────────────────────────────────────────
test_folder = os.path.join(args.dataset_path, 'testing', 'frames')
label_folder = os.path.join(args.dataset_path, 'test_label')

test_dataset = TestDataLoader(
    test_folder, label_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h, resize_width=args.w,
    num_frames=args.num_frames, dataset=args.dataset_type
)

test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers)

# ─── Storage ─────────────────────────────────────────────────────────
psnr_records = OrderedDict()
gt_records = OrderedDict()

save_img_dir = "saved_images"
save_plot_dir = "score_plots"
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_plot_dir, exist_ok=True)

print(f'Evaluating {args.dataset_type}...')
print(f"Test samples: {len(test_batch)}")

active_video = None
frame_counter = 0

# ─── Inference ───────────────────────────────────────────────────────
tic = time.time()
for k, data_dict in enumerate(test_batch):

    # if k%100!=0:
    #     continue

    imgs = data_dict['batch'].to(device)
    gt_label = data_dict['label'].item()
    video_name = data_dict['video_name'][0]

    if video_name not in psnr_records:
        psnr_records[video_name] = []
        gt_records[video_name] = []

    with torch.no_grad():
        outputs = model(imgs)
        recon_frame = outputs['output']

        total_frames = imgs.shape[2]
        mid_idx = total_frames // 2

        recon_loss = torch.mean(
            loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])
        ).item()

    psnr_records[video_name].append(psnr(recon_loss))
    gt_records[video_name].append(gt_label)

    # Track video for saving images
    if active_video is None or active_video != video_name:
        frame_counter = 0
        active_video = video_name
    frame_counter += 1

    # Save sample images every 50 frames
    # if k%50 == 0:
    if frame_counter % 50 == 0:
        label_str = "anomaly" if gt_label == 1 else "normal"

        recon_img = (recon_frame[0, :, mid_idx].cpu().numpy() + 1) * 127.5
        recon_img = recon_img.transpose(1, 2, 0).astype(np.uint8)

        orig_img = (imgs[0, :, mid_idx].cpu().numpy() + 1) * 127.5
        orig_img = orig_img.transpose(1, 2, 0).astype(np.uint8)

        diff = cv2.absdiff(orig_img, recon_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(save_img_dir, f"{video_name}_f{frame_counter:04d}_recon_{label_str}.png"), recon_img)
        cv2.imwrite(os.path.join(save_img_dir, f"{video_name}_f{frame_counter:04d}_orig_{label_str}.png"), orig_img)
        cv2.imwrite(os.path.join(save_img_dir, f"{video_name}_f{frame_counter:04d}_heatmap_{label_str}.png"), heatmap)

toc = time.time()

# ─── Compute AUC ─────────────────────────────────────────────────────
all_psnrs = []
all_gt = []

for vid in psnr_records.keys():
    all_psnrs += psnr_records[vid]
    all_gt += gt_records[vid]

anomaly_scores = anomaly_score_list(all_psnrs)
accuracy = AUC(anomaly_scores, np.expand_dims(1 - np.array(all_gt), 0))

print(f'\nAUC: {accuracy * 100:.2f}%')
if args.print_time:
    print(f'FPS: {len(test_batch) / (toc - tic):.2f}')

# ─── Per-video plots ─────────────────────────────────────────────────
for vid_name in psnr_records.keys():
    vid_scores = anomaly_score_list(psnr_records[vid_name])
    vid_gt = np.array(gt_records[vid_name])

    # Find anomaly regions for pink highlighting
    rect_start, rect_end = [], []
    active = False
    for i, val in enumerate(vid_gt):
        if val == 1 and not active:
            rect_start.append(i)
            active = True
        elif val == 0 and active:
            rect_end.append(i)
            active = False
    if active:
        rect_end.append(len(vid_gt) - 1)

    plt.figure(figsize=(12, 5))
    plt.plot(vid_scores, label='Anomaly Score', color='blue')
    plt.ylim(-0.05, 1.05)
    plt.title(f'Video: {vid_name}')
    plt.xlabel('Frames')
    plt.ylabel('Score')

    ax = plt.gca()
    for rs, re in zip(rect_start, rect_end):
        ax.add_patch(Rectangle((rs, 0), re - rs, 1, facecolor="pink", alpha=0.5))

    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, f"summary_plot_{vid_name}.png"))
    plt.close()

print("Evaluation finished. Summary plots saved.")