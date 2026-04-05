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
from model import EntropyLossEncap

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
parser.add_argument('--use_skip', action='store_true', help='enable DWT U-Net skip connections in decoder')

# Motion mask arguments
parser.add_argument('--motion_mask', action='store_true', help='enable motion mask for evaluation')
parser.add_argument('--block_size', type=int, default=16, help='block size for motion mask')
parser.add_argument('--mask_ratio', type=float, default=0.8, help='fraction of static blocks to mask out')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_entropy_loss_func = EntropyLossEncap().to(device)

# 1. Load Model
if args.model == 'VST':
    model = VST(mem_dim=args.mem_dim, use_skip=args.use_skip)
else:
    model = convAE()

if torch.cuda.is_available():
    model = nn.DataParallel(model).to(device)

model_dict = torch.load(args.model_dir, weights_only=False)

try:
    model.load_state_dict(model_dict['model'].state_dict(),strict=False)
except:
    model.load_state_dict(model_dict['model'],strict=False)

model.eval()
loss_func_mse = nn.MSELoss(reduction='none')

# 2. Setup Data (New Logic)
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

active_video = None # using this for saving input/output for a video
frame_counter = 0
# 3. Inference Loop
tic = time.time()
for k, data_dict in enumerate(test_batch):

    # if k%100!=0:
    #     continue

    imgs = data_dict['batch'].to(device)
    gt_label = data_dict['label'].item()
    video_name = data_dict['video_name'][0]

    # new
    img_index = data_dict['index'].to(device)    

    # Get motion mask if enabled
    if args.motion_mask:
        motion_mask_np = data_dict['motion_mask'].numpy()[0]  # (H, W) float32
        motion_mask_tensor = data_dict['motion_mask'].to(device)  # (1, H, W)

    if video_name not in psnr_records:
        psnr_records[video_name] = []
        gt_records[video_name] = []

    with torch.no_grad():
        outputs = model(imgs)
        recon_frame = outputs['output']
        att_w = outputs['att']
        recon_index = outputs['recon_index']

        # Get the temporal dimension size (dimension 2 for a B, C, D, H, W tensor)
        total_frames = imgs.shape[2] 
        mid_idx = total_frames // 2

        # Calculate MSE for the middle frame (always use original unmasked error for scoring)
        pixel_mse = loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])  # (C, H, W)
        recon_loss = torch.mean(pixel_mse).item()

        # entropy loss
        entropy_loss = tr_entropy_loss_func(att_w)

        # period loss (USING DATALOADER INDEX)
        period_loss = F.cross_entropy(recon_index, img_index)


    psnr_records[video_name].append(psnr(recon_loss))
    gt_records[video_name].append(gt_label)

    if active_video==None or active_video!=video_name:
        frame_counter=0
        active_video = video_name
    
    frame_counter+=1

    # if k%50 == 0:

    if frame_counter%50 == 0:

        label_str = "anomaly" if gt_label == 1 else "normal"

        recon_img = (recon_frame[0, :, mid_idx].cpu().detach().numpy() + 1) * 127.5
        recon_img = recon_img.transpose(1, 2, 0).astype(np.uint8) # Convert CHW to HWC

        orig_img = (imgs[0, :, mid_idx].cpu().detach().numpy() + 1) * 127.5
        orig_img = orig_img.transpose(1, 2, 0).astype(np.uint8) # Convert CHW to HWC

        # ----- compute heatmap -----
        diff = cv2.absdiff(orig_img, recon_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if args.motion_mask:
            # Zero out static regions in the heatmap
            diff_gray_masked = (diff_gray.astype(np.float32) * motion_mask_np).astype(np.uint8)
            diff_norm = cv2.normalize(diff_gray_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # apply heatmap colormap
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

        # Also save the motion mask visualization if enabled
        if args.motion_mask:
            mask_vis = (motion_mask_np * 255).astype(np.uint8)
            mask_name = f"{video_name}_f{frame_counter:04d}_motionmask_{label_str}.png"
            cv2.imwrite(os.path.join(save_img_dir, mask_name), mask_vis)

toc = time.time()

# 4. Global Score Calculation
all_psnrs = []
all_gt = []

# Normalize scores per video or globally (using anomaly_score_list helper)
for vid in psnr_records.keys():
    all_psnrs += psnr_records[vid]
    all_gt += gt_records[vid]

# anomaly_score_list usually scales PSNR to [0, 1] anomaly scores
anomaly_scores = anomaly_score_list(all_psnrs) 
accuracy = AUC(anomaly_scores, np.expand_dims(1-np.array(all_gt), 0))   # we 1-np.array(all_gt) prev with psnr

print(f'\nAUC: {accuracy*100:.2f}%')
if args.print_time:
    print(f'FPS: {len(test_batch)/(toc-tic):.2f}')

# 5. Summary Plotting (One per Video)
for vid_name in psnr_records.keys():
    vid_scores = anomaly_score_list(psnr_records[vid_name])
    vid_raw = np.array(psnr_records[vid_name])
    vid_gt = np.array(gt_records[vid_name])
    
    # Calculate Rectangle segments for pink highlighting
    rect_start, rect_end = [], []
    active = False
    for i, val in enumerate(vid_gt):
        if val == 1 and not active:
            rect_start.append(i)
            active = True
        elif val == 0 and active:
            rect_end.append(i)
            active = False
    if active: rect_end.append(len(vid_gt)-1)

    # --- Normalized anomaly score plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(vid_scores, label='Anomaly Score', color='blue')
    plt.ylim(-0.05, 1.05)
    plt.title(f'Video: {vid_name} (normalized)' + (' (motion masked)' if args.motion_mask else ''))
    plt.xlabel('Frames')
    plt.ylabel('Score')
    ax = plt.gca()
    for rs, re in zip(rect_start, rect_end):
        ax.add_patch(Rectangle((rs, 0), re-rs, 1, facecolor="pink", alpha=0.5))
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, f"summary_plot_{vid_name}.png"))
    plt.close()

    # --- Raw (unnormalized) PSNR plot ---
    plt.figure(figsize=(12, 5))
    plt.plot(vid_raw, label='Raw PSNR', color='green')
    plt.title(f'Video: {vid_name} (raw PSNR)' + (' (motion masked)' if args.motion_mask else ''))
    plt.xlabel('Frames')
    plt.ylabel('PSNR')
    ax = plt.gca()
    y_min, y_max = vid_raw.min(), vid_raw.max()
    for rs, re in zip(rect_start, rect_end):
        ax.add_patch(Rectangle((rs, y_min), re-rs, y_max-y_min, facecolor="pink", alpha=0.5))
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, f"summary_plot_raw_{vid_name}.png"))
    plt.close()

print("Evaluation finished. Summary plots saved.")

# python evaluate.py \
#     --dataset_type VAD \
#     --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01" \
#     --model VST \
#     --model_dir "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\ipad_repo\exp\log_VST_weight_recon_256\model_final.pth" \
#     --num_frames 8 \
#     --mem_dim 2000 \
#     --motion_mask \
#     --block_size 16 \
#     --mask_ratio 0.5