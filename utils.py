import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def psnrv2(mse, peak):
    # note: peak = max(I) where I ranged from 0 to 2 (considering mse is calculated when I is ranged -1 to 1)
    return 10 * math.log10(peak * peak / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)  # +1/2 probably the g() function. Normalize from 0-1. Although not exactly min() and max() value.
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result


# class TestDataLoader(Reconstruction3DDataLoader):
#     def __init__(self, video_folder, label_folder, transform, resize_height, resize_width, 
#                  num_frames=16, img_extension='.jpg', dataset='ped2'):
#         # 1. Initialize the parent class to set up video paths and frame lists
#         super(TestDataLoader, self).__init__(video_folder, transform, resize_height, resize_width, 
#                                              num_frames, img_extension, dataset)
        
#         self.label_dir = label_folder
#         self.video_labels = {}
        
#         # 2. Load labels and sync with existing video lengths
#         self.load_all_labels()

#     def load_all_labels(self):
#         """
#         Maps folder names (e.g., '01') to .npy files (e.g., '001.npy').
#         Trims mismatches between image count and label count.
#         """
#         for video_name in self.videos.keys():
#             # Map folder "01" to "001.npy"
#             label_file = f"{int(video_name):03d}.npy"
#             label_path = os.path.join(self.label_dir, label_file)

#             if os.path.exists(label_path):
#                 labels = np.load(label_path)
#                 num_frames = self.videos[video_name]['length']
#                 num_labels = len(labels)

#                 # Sync: Use the smaller of the two to ensure every frame has a label
#                 min_len = min(num_frames, num_labels)
#                 self.video_labels[video_name] = labels[:min_len]
                
#                 # Update parent dictionary if we had to trim frames to match labels
#                 if num_frames > min_len:
#                     self.videos[video_name]['frame'] = self.videos[video_name]['frame'][:min_len]
#                     self.videos[video_name]['length'] = min_len
#             else:
#                 print(f"Warning: Label file {label_path} not found for video {video_name}")

#     def __getitem__(self, index):
#         # Path of the first frame in the sequence
#         full_path = self.samples[index]
#         filename = os.path.basename(full_path)
#         video_name = os.path.basename(os.path.dirname(full_path))

#         # Determine frame index from filename (e.g., '081.jpg' -> 81)
#         # Note: If filenames start at 001, frame_idx 0 corresponds to '001.jpg'
#         frame_number = int(filename.split('.')[-2])
#         frame_idx = frame_number - 1 if self.dataset != 'shanghai' else frame_number - 1

#         # 1. Load the 16-frame clip
#         batch = []
#         for i in range(self._num_frames):
#             # Target is current frame + step
#             # parent's get_all_samples ensures this index is always valid
#             target_frame_path = self.videos[video_name]['frame'][frame_idx + i]
            
#             image = np_load_frame(target_frame_path, self._resize_height, 
#                                   self._resize_width, grayscale=True)
            
#             if self.transform is not None:
#                 batch.append(self.transform(image))

#         # 2. Extract Middle Label
#         # For num_frames=16, the middle frame is index 8 (the 9th frame)
#         middle_offset = self._num_frames // 2
#         label_idx = frame_idx + middle_offset
        
#         # Pull the specific label for this middle frame
#         if video_name in self.video_labels:
#             # clip index just in case of very short video/label mismatch
#             safe_idx = min(label_idx, len(self.video_labels[video_name]) - 1)
#             label = self.video_labels[video_name][safe_idx]
#         else:
#             label = 0

#         # 3. Format output
#         img = OrderedDict()
#         img['batch'] = np.stack(batch, axis=1) # Shape: (C, T, H, W)
#         img['label'] = label
#         img['video_name'] = video_name
#         img['index'] = label_idx * 200 // self.videos[video_name]['length']
        
#         return img