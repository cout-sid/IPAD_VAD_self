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

import time
from model import EntropyLossEncap
from tqdm.notebook import tqdm

import argparse

# python train.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\IPAD_dataset\R01" --model VST --epochs 2 --num_workers 0
# python evaluate.py --dataset_type VAD --dataset_path "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\IPAD_dataset\IPAD_dataset\R01" --model VST --model_dir "C:\Users\sidni\OwnDrive\ECE\MTP\Surveillance\Industrial\IPAD_work\ipad_repo\exp\log_VST_weight_recon_256\model_02.pth" --num_workers 0

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
parser.add_argument('--Period_Loss_Weight', type=float, default=0.00002, help='period loss weight')
parser.add_argument('--num_frames', type=int, default=8, help='number of frames in a clip')
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
                                           resize_height=args.h, resize_width=args.w, num_frames=args.num_frames, dataset=args.dataset_type, img_extension=img_extension)
print('ccccccccccccccccccccccccccccccccccccc')
print("TRAIN DATASET LOADED")
# train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
#                                                 resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, jump=args.jump, return_normal_seq=args.pseudo_anomaly_jump_inpainting > 0, img_extension=img_extension)


def batch_temporal_gradient_saliency(clips):
    """
    clips: (B, 3, D, H, W), normalized to [-1,1]
    returns: (B, 1, D, H, W) saliency normalized to [0,1]
    """
    B, C, D, H, W = clips.shape
    clips_01 = (clips + 1) / 2.0  # convert to [0,1]

    saliency = torch.zeros((B, D, H, W), device=clips.device, dtype=clips.dtype)

    # central temporal difference
    for t in range(1, D-1):
        prev_f = clips_01[:, :, t-1]
        mid_f  = clips_01[:, :, t]
        next_f = clips_01[:, :, t+1]

        grad = (torch.abs(mid_f - prev_f) + torch.abs(next_f - mid_f)) / 2.0
        grad = grad.mean(dim=1)  # mean across channels → scalar map
        saliency[:, t] = grad

    # forward/backward difference for edges
    saliency[:, 0]  = torch.abs(clips_01[:,:,0]  - clips_01[:,:,1]).mean(dim=1)
    saliency[:, -1] = torch.abs(clips_01[:,:,-1] - clips_01[:,:,-2]).mean(dim=1)

    # normalize each sample
    saliency_flat = saliency.view(B, -1)
    sal_min = saliency_flat.min(dim=1)[0].view(B,1,1,1)
    sal_max = saliency_flat.max(dim=1)[0].view(B,1,1,1)
    saliency = (saliency - sal_min) / (sal_max - sal_min + 1e-8)

    return saliency.unsqueeze(1)  # (B,1,D,H,W)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
# train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
#                                    shuffle=True, num_workers=args.num_workers, drop_last=True)

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

if args.start_epoch < args.epochs:
    if args.model=='VST':
        model = VST()
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
        model.load_state_dict(model_weight.state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])
        # model.cuda()
        model.to(device)

    epoch_mean_list=[]
    epoch_overall_list=[]
    epoch_entropy_list = []
    epoch_period_list = []

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


        patch_size = 8
        high_ratio = 0.7        # top 50%
        low_weight = 0.2        # weight for low-gradient patches


        # Wrap your DataLoader
        print(f"The length of train_batch => {len(train_batch)}")
        # progress_bar = tqdm(enumerate(train_batch), total=len(train_batch), desc="Training")
        pbar = tqdm(train_batch, desc=f"Epoch {epoch+1}", total=len(train_batch), ncols=85, file=orig_stdout)

        for j, imgs in enumerate(pbar):

            #imgs (batch_size,3,16,H,W)
            net_in = copy.deepcopy(imgs['batch'])
            # B, C, D, H, W = net_in.shape
            # Hp = H // patch_size
            # Wp = W // patch_size

            # net_in = net_in.cuda()
            net_in = net_in.to(device)
            img_index = copy.deepcopy(imgs['index'])
            # img_index = img_index.cuda()
            img_index = img_index.to(device)
            # len=batch_size index



            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False




            ########## TRAIN GENERATOR
            # net_in (batch_size,3,num_frames,H,W)
            Recon_frames = model.forward(net_in)
            outputs = Recon_frames['output']
            att_w = Recon_frames['att']
            recon_index = Recon_frames['recon_index']

            # memory entropy loss
            entropy_loss = tr_entropy_loss_func(att_w)#weight entropy loss
            entropy_loss_val = entropy_loss.item()
            loss_entropy = entropy_loss_weight * entropy_loss

            #recon loss
            # loss_mse = loss_func_mse(outputs, net_in)

            pixel_loss = loss_func_mse(outputs, net_in)  # (B,3,D,H,W)

            # weighted_pixel_loss = pixel_loss * weights_tube

            # loss_mse = weighted_pixel_loss.sum() / (weights_tube.sum() * C + 1e-8)
            loss_mse=pixel_loss

            #period loss
            loss_period = F.cross_entropy(recon_index,img_index)

            
            loss_period = loss_period * period_loss_weight



            modified_loss_mse = []
            # for b in range(args.batch_size):

            #     modified_loss_mse.append(torch.mean(loss_mse[b]))
            #     lossepoch += modified_loss_mse[-1].cpu().detach().item()
            #     # lossepoch += torch.mean
            #     losscounter += 1

            # assert len(modified_loss_mse) == loss_mse.size(0)
            # stacked_loss_mse = torch.stack(modified_loss_mse)
            # loss_recon = torch.mean(stacked_loss_mse)

            mid = pixel_loss.shape[2] // 2
            loss_recon = pixel_loss[:, :, mid, :, :].mean()

            # loss_recon = pixel_loss.mean()
            loss = loss_recon + loss_entropy + loss_period

            loss_recon_epoch += loss_recon.item()
            loss_entropy_epoch += loss_entropy.item()
            loss_period_epoch += loss_period.item()
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
            
            # if j==5:
            #     break

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
            totalloss=total_loss_epoch/losscounter

            print('MeanLoss: Reconstruction {:.9f}'.format(meanloss))
            print("Overall loss per clip per epoch: {:.9f}".format(totalloss))
            print('MeanLoss: Entropy {:.9f}'.format(mean_entropy))
            print('MeanLoss: Period {:.9f}'.format(mean_period))


            epoch_mean_list.append(round(meanloss, 9))
            epoch_entropy_list.append(round(mean_entropy, 9))
            epoch_period_list.append(round(mean_period, 9))
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
            
# print("RECONSTRUCTION MEAN FOR EPOCHS")
# print(epoch_mean_list)
# print("TOTAL LOSS FOR EPOCHS")
# print(epoch_overall_list)

# Plot Epoch vs Loss
epochs = list(range((args.start_epoch) + 1, (args.epochs) + 1))

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

# Show plot
# plt.show()

toc = time.time()
print('Training is finished')
print('Training time: ',(toc-tic))
sys.stdout = orig_stdout
f.close()



