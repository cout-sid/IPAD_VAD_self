import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from model.utils import TestDataLoader # Using your new loader
from model.autoencoder import *
from model.video_swin_transformer import *
from utils import *
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import time
import numpy as np
import scipy.io
import os
import cv2
from collections import OrderedDict
from model import EntropyLossEncap

# --- ADDED FOR DYNAMIC THRESHOLD CALCULATION ---
from sklearn.metrics import roc_curve, auc, precision_recall_curve

parser = argparse.ArgumentParser(description="STEAL Net Evaluation")
parser.add_argument('--model', type=str, default='VST', choices=['VST', 'conAE'])
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=256)
parser.add_argument('--dataset_type', type=str, default='VAD')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--print_time', action='store_true')
parser.add_argument('--num_frames', type=int, default=8, help='number of frames in a clip')
parser.add_argument('--mem_dim', type=int, default=2000, help='dimension of memory bank')

# Motion mask arguments
parser.add_argument('--motion_mask', action='store_true', help='enable motion mask for evaluation')
parser.add_argument('--block_size', type=int, default=16, help='block size for motion mask')
parser.add_argument('--mask_ratio', type=float, default=0.8, help='fraction of static blocks to mask out')
parser.add_argument('--use_skip', action='store_true', help='enable DWT U-Net skip connections in decoder')
parser.add_argument('--use_wavelet', action='store_true', help='enable wavelet in decoder')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_entropy_loss_func = EntropyLossEncap().to(device)

# 1. Load Model
if args.model == 'VST':
    model = VST(mem_dim=args.mem_dim, use_wavelet=args.use_wavelet, use_skip=args.use_skip)
else:
    model = convAE()

if torch.cuda.is_available():
    model = nn.DataParallel(model).to(device)

# Map location ensures CPU compatibility if CUDA is unavailable
model_dict = torch.load(args.model_dir, map_location=device, weights_only=False)

try:
    model.load_state_dict(model_dict['model'].state_dict(), strict=False)
except:
    model.load_state_dict(model_dict['model'], strict=False)

model.eval()
loss_func_mse = nn.MSELoss(reduction='none')

# 2. Setup Data
test_folder = os.path.join(args.dataset_path, 'testing', 'frames')
label_folder = os.path.join(args.dataset_path, 'test_label')

test_dataset = TestDataLoader(
    test_folder, label_folder, 
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h, resize_width=args.w, num_frames=args.num_frames,
    dataset=args.dataset_type,
    motion_mask=args.motion_mask, block_size=args.block_size, mask_ratio=args.mask_ratio
)

test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

# Storage for results
psnr_records = OrderedDict() # Stores list of PSNRs per video_name
gt_records = OrderedDict()   # Stores list of GT labels per video_name

# Extract model name from model_dir for unique output folders
model_filename = os.path.splitext(os.path.basename(args.model_dir))[0]  # e.g. "model_best"
save_img_dir = f"saved_images_{model_filename}"
save_plot_dir = f"score_plots_{model_filename}"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_plot_dir, exist_ok=True)

print(f'Evaluating {args.dataset_type}...')
print(f"Motion mask: {'ON' if args.motion_mask else 'OFF'}")
if args.motion_mask:
    print(f"  Block size: {args.block_size}, Mask ratio: {args.mask_ratio}")
print(f"length of test_batch: {len(test_batch)}")

active_video = None 
frame_counter = 0

# 3. Inference Loop
tic = time.time()
for k, data_dict in enumerate(test_batch):

    imgs = data_dict['batch'].to(device)
    gt_label = data_dict['label'].item()
    
    # [FIX] Properly extract video string name
    video_name = data_dict['video_name']
    if isinstance(video_name, list):
        video_name = video_name[0]

    img_index = data_dict['index'].to(device)   

    if args.motion_mask:
        # [FIX] Added to extract the 2D numpy array and prevent shape mismatch
        motion_mask_np = data_dict['motion_mask'].numpy()[0]
        motion_mask_tensor = data_dict['motion_mask'].to(device)

    if video_name not in psnr_records:
        psnr_records[video_name] = []
        gt_records[video_name] = []

    with torch.no_grad():
        outputs = model(imgs)
        recon_frame = outputs['output']
        att_w = outputs['att']
        recon_index = outputs['recon_index']

        # [FIX] Cast to int to prevent torch.Size operand errors with floor division
        total_frames = imgs.shape[2] 
        mid_idx = total_frames // 2

        pixel_mse = loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])
        recon_loss = torch.mean(pixel_mse).item()

        entropy_loss = tr_entropy_loss_func(att_w)
        period_loss = F.cross_entropy(recon_index, img_index)

    psnr_records[video_name].append(psnr(recon_loss))
    gt_records[video_name].append(gt_label)

    if active_video==None or active_video!=video_name:
        frame_counter=0
        active_video = video_name
    
    frame_counter+=1

    if frame_counter%50 == 0:
        label_str = "anomaly" if gt_label == 1 else "normal"

        recon_img = (recon_frame[0, :, mid_idx].cpu().detach().numpy() + 1) * 127.5
        recon_img = recon_img.transpose(1, 2, 0).astype(np.uint8) 

        orig_img = (imgs[0, :, mid_idx].cpu().detach().numpy() + 1) * 127.5
        orig_img = orig_img.transpose(1, 2, 0).astype(np.uint8) 

        diff = cv2.absdiff(orig_img, recon_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if args.motion_mask:
            diff_gray_masked = (diff_gray.astype(np.float32) * motion_mask_np).astype(np.uint8)
            diff_norm = cv2.normalize(diff_gray_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        recon_name = f"{video_name}_f{frame_counter:04d}_recon_{label_str}.png"
        orig_name = f"{video_name}_f{frame_counter:04d}_orig_{label_str}.png"
        heatmap_name = f"{video_name}_f{frame_counter:04d}_heatmap_{label_str}.png"

        recon_path = os.path.join(save_img_dir, recon_name)
        orig_path = os.path.join(save_img_dir, orig_name)
        heatmap_path = os.path.join(save_img_dir, heatmap_name)

        cv2.imwrite(recon_path, recon_img)
        cv2.imwrite(orig_path, orig_img)
        cv2.imwrite(heatmap_path, heatmap)

        if args.motion_mask:
            mask_vis = (motion_mask_np * 255).astype(np.uint8)
            mask_name = f"{video_name}_f{frame_counter:04d}_motionmask_{label_str}.png"
            cv2.imwrite(os.path.join(save_img_dir, mask_name), mask_vis)

toc = time.time()


# ==========================================================
# 4. Global Score Calculation & RAW Threshold (UPDATED)
# ==========================================================

all_psnrs = []
all_gt = []

for vid in psnr_records.keys():
    all_psnrs += psnr_records[vid]
    all_gt += gt_records[vid]

all_psnrs = np.array(all_psnrs)
all_gt = np.array(all_gt)

# ----------------------------------------------------------
# 🔹 Compute global stats (useful for reference/debug)
# ----------------------------------------------------------
psnr_min, psnr_max = all_psnrs.min(), all_psnrs.max()

print(f"Global PSNR min: {psnr_min:.4f}")
print(f"Global PSNR max: {psnr_max:.4f}")

# ----------------------------------------------------------
# 🔥 IMPORTANT: Use NEGATIVE PSNR for evaluation only
# (because anomaly = low PSNR → high score needed)
# ----------------------------------------------------------
scores_for_eval = -all_psnrs

# ----------------------------------------------------------
# 🔹 AUC Calculation
# ----------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(all_gt, scores_for_eval)
accuracy = auc(fpr, tpr)
print(f'\nAUC: {accuracy*100:.2f}%')

# ----------------------------------------------------------
# 🔥 Find optimal threshold (F1-score based)
# ----------------------------------------------------------
precision, recall, pr_thresholds = precision_recall_curve(all_gt, scores_for_eval)

f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
optimal_idx_f1 = np.argmax(f1_scores)

# ⚠️ Convert back to PSNR domain
raw_psnr_threshold = -pr_thresholds[optimal_idx_f1]
max_f1 = f1_scores[optimal_idx_f1]

print(f"Max F1-Score: {max_f1:.4f}")
print(f"Calculated RAW PSNR Threshold: {raw_psnr_threshold:.4f}\n")

# ----------------------------------------------------------
# 💾 Save threshold info (VERY IMPORTANT)
# ----------------------------------------------------------
threshold_save_path = f"threshold_info_{model_filename}.npy"

np.save(threshold_save_path, {
    "psnr_threshold": raw_psnr_threshold,
    "psnr_min": psnr_min,
    "psnr_max": psnr_max
})

print(f"Threshold info saved to: {threshold_save_path}")

if args.print_time:
    print(f'FPS: {len(test_batch)/(toc-tic):.2f}')