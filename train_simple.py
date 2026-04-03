"""
train_simple.py

Minimal training script for SimpleVST_AE.
Only MSE reconstruction loss on the middle frame (or all frames).

Usage:
  python train_simple.py \
    --dataset_path /path/to/R01 \
    --dataset_type VAD \
    --epochs 50 \
    --batch_size 8 \
    --num_frames 16 \
    --lr 1e-4
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import random_split

from model.utils import Reconstruction3DDataLoader
from model.simple_vst_ae import SimpleVST_AE
from utils import *

import time
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


# python train_simple.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01"  --epochs 2 --num_workers 0 
# python evaluate_simple.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01"  --model_dir "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\ipad_repo\exp\log_simple_vst\model_02.pth" --num_workers 0

# ─── Args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simple VST AE Training")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--dataset_type', type=str, default='VAD')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--exp_dir', type=str, default='log_simple_vst')
parser.add_argument('--model_dir', type=str, default=None, help='path to resume from')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--num_frames', type=int, default=8)
parser.add_argument('--all_frame_error', action='store_true',
                    help='use all frames for loss instead of just middle frame')
parser.add_argument('--print_all', action='store_true')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── Experiment directory ────────────────────────────────────────────
log_dir = os.path.join('./exp', args.exp_dir)
os.makedirs(log_dir, exist_ok=True)

print("-" * 50)
print(f"{'Training Arguments':^50}")
print("-" * 50)
for key, value in vars(args).items():
    print(f"{key:<25}: {value}")
print("-" * 50)

# ─── Dataset ─────────────────────────────────────────────────────────
train_folder = os.path.join(args.dataset_path, 'training', 'frames')
img_extension = '.jpg'

train_dataset = Reconstruction3DDataLoader(
    train_folder,
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h, resize_width=args.w,
    num_frames=args.num_frames, dataset=args.dataset_type,
    img_extension=img_extension
)

# Train/Val split
val_ratio = 0.10
val_size = int(len(train_dataset) * val_ratio)
train_size = len(train_dataset) - val_size

train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_batch = data.DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
val_batch = data.DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

print(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")

# ─── Model ───────────────────────────────────────────────────────────
model = SimpleVST_AE()

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Using {device_count} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model.cuda()
else:
    print("Using CPU.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_func_mse = nn.MSELoss(reduction='none')

# Resume
if args.model_dir is not None:
    assert args.start_epoch > 0
    model_dict = torch.load(args.model_dir, weights_only=False)
    model.load_state_dict(model_dict['model'].state_dict())
    optimizer.load_state_dict(model_dict['optimizer'])
    model.to(device)

# ─── Training loop ───────────────────────────────────────────────────
epoch_loss_list = []
tic = time.time()

for epoch in range(args.start_epoch, args.epochs):
    model.train()
    loss_epoch = 0
    count = 0

    pbar = tqdm(train_batch, desc=f"Epoch {epoch+1}/{args.epochs}",
                total=len(train_batch), ncols=90)

    for j, imgs in enumerate(pbar):
        net_in = imgs['batch'].to(device)
        # net_in: (B, 3, 16, 256, 256)

        outputs = model(net_in)
        recon = outputs['output']

        # Compute loss
        pixel_loss = loss_func_mse(recon, net_in)

        if args.all_frame_error:
            loss = pixel_loss.mean()
        else:
            mid = pixel_loss.shape[2] // 2
            loss = pixel_loss[:, :, mid, :, :].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if j==5:
        #     break

        loss_epoch += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if j % 100 == 0 or args.print_all:
            print(f"  epoch {epoch+1} iter {j}/{len(train_batch)} | loss: {loss.item():.6f}")

    avg_loss = loss_epoch / count
    epoch_loss_list.append(avg_loss)
    print(f"Epoch {epoch+1} | Avg Recon Loss: {avg_loss:.9f}")

    # Save checkpoint
    model_dict = {'model': model, 'optimizer': optimizer.state_dict()}
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        torch.save(model_dict, os.path.join(log_dir, f'model_{epoch+1:02d}.pth'))

    if epoch == args.epochs-1:
        torch.save(model_dict, os.path.join(log_dir, 'model_final.pth'))

# ─── Save loss plot ──────────────────────────────────────────────────
toc = time.time()
print(f"\nTraining finished in {toc-tic:.1f}s")

epochs_list = list(range(args.start_epoch + 1, args.start_epoch + len(epoch_loss_list) + 1))
plt.figure()
plt.plot(epochs_list, epoch_loss_list, label="Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(log_dir, "loss_vs_epoch.png"))
print(f"Loss plot saved to {log_dir}/loss_vs_epoch.png")

# Save losses to CSV
df = pd.DataFrame({'epoch': epochs_list, 'recon_loss': epoch_loss_list})
df.to_csv(os.path.join(log_dir, 'training_losses.csv'), index=False)
print(f"Losses saved to {log_dir}/training_losses.csv")

# ─── Quick validation on a few training samples ──────────────────────
validate_indices = [50, 100, 200, 500]
validate_dir = os.path.join(log_dir, "validate_images")
os.makedirs(validate_dir, exist_ok=True)

model.eval()
for idx in validate_indices:
    if idx >= len(train_subset):
        continue
    sample = train_subset[idx]
    imgs = torch.tensor(sample['batch']).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(imgs)['output']

    mid_idx = imgs.shape[2] // 2

    orig_img = (imgs[0, :, mid_idx].cpu().numpy() + 1) * 127.5
    orig_img = orig_img.transpose(1, 2, 0).astype(np.uint8)

    recon_img = (recon[0, :, mid_idx].cpu().numpy() + 1) * 127.5
    recon_img = recon_img.transpose(1, 2, 0).astype(np.uint8)

    diff = cv2.absdiff(orig_img, recon_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_original.png"), orig_img)
    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_recon.png"), recon_img)
    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_heatmap.png"), heatmap)

print(f"Validation images saved to {validate_dir}/")