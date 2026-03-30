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


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_entropy_loss_func = EntropyLossEncap().to(device)

# 1. Load Model
if args.model == 'VST':
    model = VST(mem_dim=args.mem_dim)
else:
    model = convAE()

if torch.cuda.is_available():
    model = nn.DataParallel(model).to(device)

model_dict = torch.load(args.model_dir, weights_only=False)

try:
    model.load_state_dict(model_dict['model'].state_dict())
except:
    model.load_state_dict(model_dict['model'])

model.eval()
loss_func_mse = nn.MSELoss(reduction='none')

# 2. Setup Data (New Logic)
test_folder = os.path.join(args.dataset_path, 'testing', 'frames')
label_folder = os.path.join(args.dataset_path, 'test_label')

test_dataset = TestDataLoader(
    test_folder, label_folder, 
    transforms.Compose([transforms.ToTensor()]),
    resize_height=args.h, resize_width=args.w, num_frames=args.num_frames,
    dataset=args.dataset_type
)

test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

# Storage for results
psnr_records = OrderedDict() # Stores list of PSNRs per video_name
gt_records = OrderedDict()   # Stores list of GT labels per video_name

save_img_dir = "saved_images"
save_plot_dir = "score_plots"
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_plot_dir, exist_ok=True)

print(f'Evaluating {args.dataset_type}...')
print(f"length of test_batch: {len(test_batch)}")

active_video = None # using this for saving input/output for a video
frame_counter = 0
# 3. Inference Loop
tic = time.time()
for k, data_dict in enumerate(test_batch):

    # if k%100!=0:
        # continue

    imgs = data_dict['batch'].to(device)
    gt_label = data_dict['label'].item()
    video_name = data_dict['video_name'][0]

    # new
    img_index = data_dict['index'].to(device)    

    if video_name not in psnr_records:
        psnr_records[video_name] = []
        gt_records[video_name] = []

    with torch.no_grad():
        outputs = model(imgs)
        recon_frame = outputs['output']
        att_w = outputs['att']
        recon_index = outputs['recon_index']
        motion_mask = outputs['motion_mask']

        
        # Get the temporal dimension size (dimension 2 for a B, C, D, H, W tensor)
        total_frames = imgs.shape[2] 
        mid_idx = total_frames // 2

        mask_mid = motion_mask[0, 0, 0].cpu().numpy()  # (H, W)

        pixel_mse = loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])
        weighted_mse = pixel_mse.mean(dim=0) * torch.tensor(mask_mid).to(device)
        recon_loss = weighted_mse.mean().item()

        # Calculate MSE for the middle frame  
        # OLD ONE
        # recon_loss = torch.mean(loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])).item()

        # entropy loss
        entropy_loss = tr_entropy_loss_func(att_w)

        # period loss (USING DATALOADER INDEX)
        period_loss = F.cross_entropy(recon_index, img_index)

        # entropy 0.0002  period 0.02

        # total_loss = recon_loss + (0.0002)*entropy_loss + (0.02)*period_loss


    psnr_records[video_name].append(psnr(recon_loss)) # replaced psnr(recon_loss) with total loss.item()
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

        # convert to grayscale difference
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # diff_weighted = diff_gray * mask_mid
        diff_weighted = diff_gray.astype(np.float32) * mask_mid
        diff_norm = cv2.normalize(diff_weighted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # normalize for visualization
        # diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

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
    # anomaly_score_list normalizes the scores for plotting 
    # this function doesn't calculate psnr

    # USING  INV FUNCTION  MAKES SCORES HIGH FOR ANOMALIES AND LOW FOR NORMAL

    vid_scores = anomaly_score_list(psnr_records[vid_name])
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

    plt.figure(figsize=(12, 5))
    plt.plot(vid_scores, label='Anomaly Score', color='blue')
    plt.ylim(-0.05, 1.05)
    plt.title(f'Video: {vid_name}')
    plt.xlabel('Frames')
    plt.ylabel('Score')
    
    # Draw Pink Rectangles
    ax = plt.gca()
    for rs, re in zip(rect_start, rect_end):
        ax.add_patch(Rectangle((rs, 0), re-rs, 1, facecolor="pink", alpha=0.5))
    
    plt.legend()
    plot_name = f"summary_plot_{vid_name}.png"
    summary_plot_path = os.path.join(save_plot_dir,plot_name)
    plt.savefig(summary_plot_path)
    plt.close()

print("Evaluation finished. Summary plots saved.")
