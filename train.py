import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets import ImageFolder
from model.utils import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump
from model.autoencoder import *
from model.video_swin_transformer import *
from utils import *
# from model.pseudoanomaly_utils import create_pseudoanomaly_cifar_smooth, \
#     create_pseudoanomaly_cifar_smoothborder, create_pseudoanomaly_seq_smoothborder, \
#     create_pseudoanomaly_cifar_cutmix, create_pseudoanomaly_cifar_mixupcutmix
from torch.utils.data import random_split

import time
from model import EntropyLossEncap
from tqdm.notebook import tqdm
import pandas as pd
from pytorch_msssim import ssim


import argparse

# python train.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01" --model VST --epochs 2 --num_workers 0 --mem_dim 2000 --motion_mask --block_size 16 --mask_ratio 0.5
# python evaluate.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\ipad_half_video\R01" --model VST --model_dir "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\ipad_repo\exp\log_VST_weight_recon_256\model_02.pth" --num_workers 0 --mem_dim 2000 --motion_mask --block_size 16 --mask_ratio 0.5

parser = argparse.ArgumentParser(description="STEAL Net")
parser.add_argument('--model', type=str, default='VST', choices=['VST','conAE'])
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate phase 1')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2','avenue', 'shanghai','SW_video','VAD','IPAD'], help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')

parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# related to skipping frame pseudo anomaly
parser.add_argument('--pseudo_anomaly_jump_inpainting', type=float, default=0, help='pseudo anomaly jump frame (skip frame) probability but with inpainting-like loss. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[3], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3

# related to patch based pseudo anomaly
parser.add_argument('--pseudo_anomaly_cifar_inpainting_smooth', type=float, default=0, help='pseudo anomaly using cifar100 patch (SmoothMixC) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_shanghai_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using shanghai patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_ped2_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using ped2 patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_SW_video_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using SW_video patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_VAD_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using VAD patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using cifar100 patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_cutmix', type=float, default=0, help='pseudo anomaly using cifar100 patch (CutMix) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size hyperparameter')
parser.add_argument('--pseudo_anomaly_imagenet_inpainting_smoothborder', type=float, default=0, help='pseudo anomaly using imagenet patch (SmoothMixS) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size (as max sigma) hyperparameter')
parser.add_argument('--pseudo_anomaly_cifar_inpainting_mixupcutmix', type=float, default=0, help='pseudo anomaly using cifar100 patch (MixUp-patch) but the loss is using inpainting-like loss. 0 no pseudo anomaly. also using max_size hyperparameter')
parser.add_argument('--max_size', type=float, default=0.2, help='maximum size of the patch relative to the input (hyperparameter alpha)')
parser.add_argument('--max_move', type=int, default=0, help='maximum movement in pixel of the patch to the input (hyperparameter beta)')

parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')
parser.add_argument('--Entropy_Loss_Weight', type=float, default=0.00002, help='entropy loss weight')
parser.add_argument('--Period_Loss_Weight', type=float, default=0.002, help='period loss weight')
parser.add_argument('--num_frames', type=int, default=8, help='number of frames in a clip')
parser.add_argument('--mem_dim', type=int, default=2000, help='dimension of memory bank')
parser.add_argument('--all_frame_error', action='store_true', help='whether to use whole batch or only mid frame for error')

parser.add_argument('--motion_mask', action='store_true', help='use motion mask for loss')
parser.add_argument('--block_size', type=int, default=32, help='block size for motion mask')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='fraction of static blocks to mask')
parser.add_argument('--use_skip', action='store_true', help='enable DWT U-Net skip connections in decoder')
parser.add_argument('--use_wavelet', action='store_true', help='enable wavelet in decoder')

##################

args = parser.parse_args()
entropy_loss_weight = args.Entropy_Loss_Weight
period_loss_weight = args.Period_Loss_Weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tr_entropy_loss_func = EntropyLossEncap().to(device)
# assert 1 not in args.jump

exp_dir = args.exp_dir
exp_dir += 'lr' + str(args.lr) if args.lr != 1e-4 else ''
exp_dir += '_'
exp_dir += args.model
exp_dir += '_weight'
exp_dir += '_recon_256'



# 2. Print them beautifully
print("-" * 50)
print(f"{'Training Arguments':^50}")
print("-" * 50)
for key, value in vars(args).items():
    print(f"{key:<25}: {value}")
print("-" * 50)



print('exp_dir: ', exp_dir)

# torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
# FIX ME FIX ME

# train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')
train_folder = os.path.join(args.dataset_path, 'training', 'frames')


# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
print('ccccccccccccccccccccccccccccccccccccc')
print("BEFORE TRAIN DATASET")
train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                           resize_height=args.h, resize_width=args.w, num_frames=args.num_frames, dataset=args.dataset_type,
                                             img_extension=img_extension,motion_mask=args.motion_mask, block_size=args.block_size, mask_ratio=args.mask_ratio)
print('ccccccccccccccccccccccccccccccccccccc')
print("TRAIN DATASET LOADED")

# train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
#                                                 resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, jump=args.jump, return_normal_seq=args.pseudo_anomaly_jump_inpainting > 0, img_extension=img_extension)







# --- Train/Val split for early stopping ---


val_ratio = 0.10
val_size = int(len(train_dataset) * val_ratio)
train_size = len(train_dataset) - val_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))

train_batch = data.DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

val_batch = data.DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

print(f"Train: {len(train_subset)} samples, Val: {len(val_subset)} samples")





# Report the training process
log_dir = os.path.join('./exp', exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'a')
# sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

tic = time.time()

epochs_ran = 0

if args.start_epoch < args.epochs:
    if args.model=='VST':
        model = VST(mem_dim=args.mem_dim,use_wavelet=args.use_wavelet, use_skip=args.use_skip)
    else:
        model = convAE()
    
    if torch.cuda.is_available():
    # Check how many GPUs are actually available
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Using {device_count} GPUs with DataParallel")
            model = nn.DataParallel(model)
        else:
            print("Using 1 GPU (No DataParallel)")
    
        model.cuda() # Moves model to GPU
    else:
        print("No CUDA detected. Using CPU.")
        model.cpu() # Moves model to CPU

    # model = nn.DataParallel(model)
    # model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume
    if args.model_dir is not None:
        assert args.start_epoch > 0
        # Loading the trained model
        model_dict = torch.load(args.model_dir, weights_only=False)
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict(),strict=False)
        optimizer.load_state_dict(model_dict['optimizer'])
        # model.cuda()
        model.to(device)

    epoch_mean_list=[]
    epoch_overall_list=[]
    epoch_entropy_list = []
    epoch_period_list = []
    epoch_ssim_list = []
    epoch_val_list = []

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        print(epoch+1)
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        loss_recon_epoch = 0
        total_loss_epoch = 0
        loss_entropy_epoch = 0
        loss_period_epoch = 0
        loss_ssim_epoch = 0




        # Wrap your DataLoader
        print(f"The length of train_batch => {len(train_batch)}")
        # progress_bar = tqdm(enumerate(train_batch), total=len(train_batch), desc="Training")
        pbar = tqdm(train_batch, desc=f"Epoch {epoch+1}", total=len(train_batch), ncols=85, file=orig_stdout)

        for j, imgs in enumerate(pbar):

            net_in = imgs['batch'].to(device)
            img_index = imgs['index'].to(device)

            ########## APPLY BINARY MASK TO INPUT (MAE-style inpainting)
            # net_in (batch_size, 3, num_frames, H, W)
            if args.motion_mask:
                mask = imgs['motion_mask'].to(device)       # (B, H, W) binary: 0=static, 1=motion
                mask_expanded = mask.unsqueeze(1)            # (B, 1, H, W)
                # Expand mask across channels and all temporal frames
                mask_tube = mask_expanded.unsqueeze(2)       # (B, 1, 1, H, W)
                # Keep original unmasked input for loss computation
                net_in_original = net_in.clone()
                # Mask the input: zero out static regions across the whole clip
                net_in = net_in * mask_tube                  # (B, 3, T, H, W) * (B, 1, 1, H, W)

            ########## TRAIN GENERATOR
            Recon_frames = model(net_in)
            outputs = Recon_frames['output']
            att_w = Recon_frames['att']
            recon_index = Recon_frames['recon_index']

            # Loss is computed against ORIGINAL unmasked input
            if args.motion_mask:
                pixel_loss = loss_func_mse(outputs, net_in_original)  # (B,3,D,H,W)
            else:
                pixel_loss = loss_func_mse(outputs, net_in)           # (B,3,D,H,W)


            # memory entropy loss
            entropy_loss = tr_entropy_loss_func(att_w)#weight entropy loss
            loss_entropy = entropy_loss_weight * entropy_loss
            # loss_entropy = torch.tensor(0.0, device=device)
            

            #period loss
            loss_period = F.cross_entropy(recon_index,img_index)
            loss_period = loss_period * period_loss_weight
            # loss_period=torch.tensor(0.0, device=device)




            mid = pixel_loss.shape[2] // 2


            if not args.all_frame_error:
                # Middle frame only — full MSE (no mask weighting needed, model must reconstruct everything)
                loss_recon = pixel_loss[:, :, mid, :, :].mean()

                # out_flat = outputs.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                # inp_flat = net_in.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                # loss_ssim = 1 - ssim(out_flat, inp_flat, data_range=2.0, win_size=5, size_average=True)
                loss_ssim = torch.tensor(0.0, device=device)

            else:
                # All frames — full MSE
                loss_recon = pixel_loss.mean()

                loss_ssim = torch.tensor(0.0, device=device)

            

            loss = loss_recon + loss_entropy + loss_period
            



            loss_recon_epoch += loss_recon.item()
            loss_entropy_epoch += loss_entropy.item()
            loss_period_epoch += loss_period.item()
            loss_ssim_epoch+= loss_ssim.item()
            total_loss_epoch += loss.item()

            losscounter += 1

            # print('Loss: {:.6f}, Loss_recon: {:.6f}, Loss_entropy: {:.6f}'.format(loss.item(),loss_recon.item(),loss_entropy.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 100 == 0 or args.print_all:
                print("epoch {:d} iter {:d}/{:d}".format(epoch+1, j, len(train_batch)))
                print('Loss: {:.6f}'.format(loss.item()))
                print('Loss: {:.6f}, Loss_recon: {:.6f}, Loss_entropy: {:.6f}, Loss_period: {:.6f}'.format(loss.item(),loss_recon.item(),loss_entropy.item(),loss_period.item()))
                print('Loss_ssim: {:.6f}'.format(loss_ssim.item()))
            
            if j==5:
            #     if args.motion_mask:
            #         mask_np = mask[0].cpu().numpy()
            #         orig = (net_in[0, :, mid].cpu().numpy() + 1) * 127.5
            #         orig = orig.transpose(1, 2, 0).astype(np.uint8)
            #         masked_img = (orig.astype(np.float32) * np.stack([mask_np]*3, axis=-1)).astype(np.uint8)
            #         cv2.imwrite(f"masked_sample_{epoch}.png", masked_img)
            #         print(f"Saved masked_sample_{epoch}.png")
                break
                

            pbar.set_postfix(batch=j)

        print('----------------------------------------')
        print('Epoch:', epoch+1)
        # if pseudolosscounter != 0:
        #     print('PseudoMeanLoss: Reconstruction {:.9f}'.format(pseudolossepoch/pseudolosscounter))
        if losscounter != 0:
            # print('MeanLoss: Reconstruction {:.9f}'.format(lossepoch/losscounter))
            meanloss=loss_recon_epoch/losscounter
            mean_entropy = loss_entropy_epoch / losscounter
            mean_period = loss_period_epoch / losscounter
            mean_ssim_epoch = loss_ssim_epoch/losscounter

            totalloss=total_loss_epoch/losscounter

            print('MeanLoss: Reconstruction {:.9f}'.format(meanloss))
            print("Overall loss per clip per epoch: {:.9f}".format(totalloss))
            print('MeanLoss: Entropy {:.9f}'.format(mean_entropy))
            print('MeanLoss: Period {:.9f}'.format(mean_period))
            print('SSIMLoss:  {:.9f}'.format(mean_ssim_epoch))



            epoch_mean_list.append(round(meanloss, 9))
            epoch_entropy_list.append(round(mean_entropy, 9))
            epoch_period_list.append(round(mean_period, 9))
            epoch_ssim_list.append(round(mean_ssim_epoch,9))
            epoch_overall_list.append(round(totalloss, 9))






        # Save the model and the memory items
        model_dict = {
            'model': model,
            'optimizer': optimizer.state_dict(),
        }

        if (epoch+1)%5 == 0 or epoch == args.epochs-1:
            torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch+1)))

        if epoch == args.epochs-1:
            torch.save(model_dict, os.path.join(log_dir, 'model_final.pth'))

        # ---------------------------------------------------------
        # --- Validation (no early stopping, just tracking) ---
        # ---------------------------------------------------------

        epochs_ran = epoch + 1

        model.eval()
        val_loss_total = 0
        val_count = 0

        with torch.no_grad():
            for val_imgs in val_batch:
                val_in = val_imgs['batch'].to(device)

                # Apply same masking as training if motion_mask is enabled
                if args.motion_mask:
                    val_mask = val_imgs['motion_mask'].to(device)        # (B, H, W)
                    val_mask_tube = val_mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, H, W)
                    val_in_original = val_in.clone()
                    val_in = val_in * val_mask_tube

                val_out = model(val_in)
                val_recon = val_out['output']

                # Compute loss same way as training
                if args.motion_mask:
                    val_pixel_loss = loss_func_mse(val_recon, val_in_original)
                else:
                    val_pixel_loss = loss_func_mse(val_recon, val_in)

                if not args.all_frame_error:
                    val_mid = val_pixel_loss.shape[2] // 2
                    val_mse = val_pixel_loss[:, :, val_mid, :, :].mean().item()
                else:
                    val_mse = val_pixel_loss.mean().item()

                val_loss_total += val_mse
                val_count += 1

        val_loss_avg = val_loss_total / val_count
        epoch_val_list.append(round(val_loss_avg, 9))
        print(f'Validation Recon Loss: {val_loss_avg:.9f}')

        model.train()
            
# print("RECONSTRUCTION MEAN FOR EPOCHS")
# print(epoch_mean_list)
# print("TOTAL LOSS FOR EPOCHS")
# print(epoch_overall_list)

# Plot Epoch vs Loss




#-----------------------------------------
#  validation part
# -----------------------------------------




# --- Validate reconstruction on specific training samples ---
validate_indices = [ 50, 100, 200, 500, 600,650,800]  # change these to whatever you want
validate_dir =  "validate_images"
os.makedirs(validate_dir, exist_ok=True)

model.eval()
loss_func_mse_val = nn.MSELoss(reduction='none')

print("\n--- Validation: Reconstructing training samples ---")

for idx in validate_indices:
    if idx >= len(train_subset):
        print(f"  Skipping index {idx} (dataset has {len(train_subset)} samples)")
        continue

    sample = train_subset[idx]
    imgs = torch.tensor(sample['batch']).unsqueeze(0).to(device)  # (1, 3, 8, 256, 256)

    with torch.no_grad():
        outputs = model(imgs)
        recon_frame = outputs['output']
        # motion_mask = outputs['motion_mask']


    mid_idx = imgs.shape[2] // 2

    # mask_mid = motion_mask[0, 0, 0].cpu().numpy()  # (H, W)
    
#     # Weighted MSE for scoring
#     pixel_mse = loss_func_mse(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])
#     weighted_mse = pixel_mse.mean(dim=0) * torch.tensor(mask_mid).to(device)
#     recon_loss = weighted_mse.mean().item()


    # MSE
    # mask_tensor = motion_mask[0, :, 0, :, :]
    # mse_val = loss_func_mse_val(recon_frame[0, :, mid_idx], imgs[0, :, mid_idx])
    # weighted_mse = (mse_val * mask_tensor).mean().item()  # .item() moves scalar to CPU

    # print(f"  Sample {idx}: Weighted MSE = {weighted_mse:.8f}")
    # print(f"  Sample {idx}: MSE = {mse_val:.8f}")

    # Original
    orig_img = (imgs[0, :, mid_idx].cpu().numpy() + 1) * 127.5
    orig_img = orig_img.transpose(1, 2, 0).astype(np.uint8)

    # Reconstruction
    recon_img = (recon_frame[0, :, mid_idx].cpu().numpy() + 1) * 127.5
    recon_img = recon_img.transpose(1, 2, 0).astype(np.uint8)

    # Heatmap
    diff = cv2.absdiff(orig_img, recon_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # diff_weighted = diff_gray.astype(np.float32) * mask_mid

    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_original.png"), orig_img)
    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_recon.png"), recon_img)
    cv2.imwrite(os.path.join(validate_dir, f"sample{idx}_heatmap.png"), heatmap)

model.train()
print(f"Validation images saved to {validate_dir}/")



# 1. Combine your lists into a dictionary
loss_data = {
    'epoch': range(1, len(epoch_mean_list) + 1),
    'mean_loss': epoch_mean_list,
    'entropy_loss': epoch_entropy_list,
    'period_loss': epoch_period_list,
    'ssim_loss': epoch_ssim_list,
    'overall_loss': epoch_overall_list,
    'val_loss': epoch_val_list
}

# 2. Convert to a DataFrame and save
df = pd.DataFrame(loss_data)
df.to_csv('training_losses.csv', index=False)
print("*"*50)
print("Saved losses to training_losses.csv")
print("*"*50)


epochs = list(range((args.start_epoch) + 1, (epochs_ran) + 1))

plt.figure()
plt.plot(epochs, epoch_mean_list, label="Reconstruction Loss")
plt.plot(epochs, epoch_entropy_list, label="Entropy Loss (weighted)")
plt.plot(epochs, epoch_period_list, label="Period Loss (weighted)")
plt.plot(epochs, epoch_overall_list, label="Total Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig(os.path.join(log_dir, "loss_vs_epoch.png"))
plt.close()

# --- Train vs Val plot in separate folder ---
train_val_plot_dir = os.path.join(log_dir, 'train_vs_val_plot')
os.makedirs(train_val_plot_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_val_list, label='Val Recon Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Recon Loss')
plt.title('Validation Reconstruction Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(train_val_plot_dir, 'val_loss.png'))
plt.close()
print(f"Train vs Val plot saved to {train_val_plot_dir}/")

toc = time.time()
print('Training is finished')
print('Training time: ',(toc-tic))
sys.stdout = orig_stdout
f.close()

