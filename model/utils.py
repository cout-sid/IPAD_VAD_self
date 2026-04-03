import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random
from PIL import Image


rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, grayscale=False):
    grayscale = False
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if grayscale:
        image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = np.copy(image_decoded)
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized
# shape => (h,w,3)


def compute_motion_mask(frames, mid_idx, block_size=16, mask_ratio=0.5):
    """
    Compute a binary motion mask for the middle frame using prev and next frames.
    Blocks with LOW motion (static regions) are masked OUT (set to 0).
    Blocks with HIGH motion are kept (set to 1).
    
    Args:
        frames: list of numpy arrays (H, W, 3) in [-1, 1] range
        mid_idx: index of the middle frame in the list
        block_size: size of blocks for motion scoring (default 16)
        mask_ratio: fraction of lowest-motion blocks to mask out (default 0.5)
    
    Returns:
        mask: numpy array (H, W) with values 0 (masked/static) or 1 (keep/motion)
    """
    mid_frame = frames[mid_idx]
    h, w = mid_frame.shape[:2]
    
    # Convert from [-1, 1] to [0, 255] uint8 for difference computation
    def to_uint8(f):
        return ((f + 1.0) * 127.5).astype(np.uint8)
    
    mid_uint8 = to_uint8(mid_frame)
    mid_gray = cv2.cvtColor(mid_uint8, cv2.COLOR_BGR2GRAY)
    
    # Use both prev and next frame if available
    diff_accum = np.zeros_like(mid_gray, dtype=np.float32)
    count = 0
    
    if mid_idx > 0:
        prev_gray = cv2.cvtColor(to_uint8(frames[mid_idx - 1]), cv2.COLOR_BGR2GRAY)
        diff_accum += cv2.absdiff(mid_gray, prev_gray).astype(np.float32)
        count += 1
    
    if mid_idx < len(frames) - 1:
        next_gray = cv2.cvtColor(to_uint8(frames[mid_idx + 1]), cv2.COLOR_BGR2GRAY)
        diff_accum += cv2.absdiff(mid_gray, next_gray).astype(np.float32)
        count += 1
    
    if count > 0:
        diff_accum /= count  # average of prev and next differences
    
    # Crop to be divisible by block_size
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    diff_cropped = diff_accum[:h_crop, :w_crop]
    
    # Score each block
    num_blocks_h = h_crop // block_size
    num_blocks_w = w_crop // block_size
    total_blocks = num_blocks_h * num_blocks_w
    
    block_scores = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y_s = i * block_size
            y_e = y_s + block_size
            x_s = j * block_size
            x_e = x_s + block_size
            score = np.sum(diff_cropped[y_s:y_e, x_s:x_e])
            block_scores.append((score, i, j))
    
    # Sort ascending (lowest motion first)
    block_scores.sort(key=lambda x: x[0])
    
    # Mask out the lowest-motion blocks
    num_to_mask = int(total_blocks * mask_ratio)
    masked_blocks = set()
    for k in range(num_to_mask):
        _, bi, bj = block_scores[k]
        masked_blocks.add((bi, bj))
    
    # Build mask (full image size, 0 = masked/static, 1 = keep/motion)
    mask = np.ones((h, w), dtype=np.float32)
    for (bi, bj) in masked_blocks:
        y_s = bi * block_size
        y_e = y_s + block_size
        x_s = bj * block_size
        x_e = x_s + block_size
        mask[y_s:y_e, x_s:x_e] = 0.0
    
    return mask


class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=16,
                 img_extension='.jpg', dataset='ped2', jump=[2], hold=[2], return_normal_seq=False,
                 motion_mask=False, block_size=16, mask_ratio=0.5):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames

        self.extension = img_extension
        self.dataset = dataset

        # Motion mask params
        self.motion_mask = motion_mask
        self.block_size = block_size
        self.mask_ratio = mask_ratio

        self.setup()
        self.samples, self.background_models = self.get_all_samples()

        self.jump = jump
        self.hold = hold
        self.return_normal_seq = return_normal_seq  # for fast and slow moving

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in sorted(videos):
            print(video)

            # video_name = video.split('\\')[-2]
            video_name = os.path.basename(os.path.normpath(video))
            
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.extension))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        background_models = []
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in sorted(videos):
            # video_name = video.split('\\')[-2]
            video_name = os.path.basename(os.path.normpath(video))

            for i in range(len(self.videos[video_name]['frame']) - self._num_frames + 1):
                frames.append(self.videos[video_name]['frame'][i])
                # background_models.append(bg_filename)

        return frames, background_models

    def __getitem__(self, index):
        filename = os.path.basename(self.samples[index])

        parent_dir = os.path.dirname(self.samples[index])
        video_name = os.path.basename(parent_dir)

        frame_number_str = filename.split('.')[-2]

        if self.dataset == 'shanghai' and 'training' in self.samples[index]:
            frame_name = int(frame_number_str) - 1
        else:
            frame_name = int(frame_number_str)

        batch = []
        raw_frames = []  # store raw numpy frames for motion mask
        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width, grayscale=False)
            
            if self.motion_mask:
                raw_frames.append(image)  # (H, W, 3) in [-1, 1]
            
            if self.transform is not None:
                batch.append(self.transform(image))

        img = OrderedDict()
        img['batch'] = np.stack(batch, axis=1)
        img['index'] = frame_name*200//len(self.videos[video_name]['frame'])
        
        # Compute motion mask if enabled
        if self.motion_mask:
            mid_idx = self._num_frames // 2
            mask = compute_motion_mask(raw_frames, mid_idx, 
                                       self.block_size, self.mask_ratio)
            img['motion_mask'] = mask  # (H, W) float32, 0=static 1=motion
        
        return img

    def __len__(self):
        return len(self.samples)
    

class TestDataLoader(Reconstruction3DDataLoader):
    def __init__(self, video_folder, label_folder, transform, resize_height, resize_width, 
                 num_frames=16, img_extension='.jpg', dataset='ped2',
                 motion_mask=False, block_size=16, mask_ratio=0.5):
        # Initialize parent with motion mask params
        super(TestDataLoader, self).__init__(video_folder, transform, resize_height, resize_width, 
                                             num_frames, img_extension, dataset,
                                             motion_mask=motion_mask, block_size=block_size,
                                             mask_ratio=mask_ratio)
        
        self.label_dir = label_folder
        self.video_labels = {}
        
        self.load_all_labels()

    def load_all_labels(self):
        """
        Maps folder names (e.g., '01') to .npy files (e.g., '001.npy').
        Trims mismatches between image count and label count.
        """
        for video_name in self.videos.keys():
            # Map folder "01" to "001.npy"
            label_file = f"{int(video_name):03d}.npy"
            label_path = os.path.join(self.label_dir, label_file)

            if os.path.exists(label_path):
                labels = np.load(label_path)
                num_frames = self.videos[video_name]['length']
                num_labels = len(labels)

                min_len = min(num_frames, num_labels)
                self.video_labels[video_name] = labels[:min_len]
                
                if num_frames > min_len:
                    self.videos[video_name]['frame'] = self.videos[video_name]['frame'][:min_len]
                    self.videos[video_name]['length'] = min_len
            else:
                print(f"Warning: Label file {label_path} not found for video {video_name}")

    def __getitem__(self, index):
        full_path = self.samples[index]
        filename = os.path.basename(full_path)
        video_name = os.path.basename(os.path.dirname(full_path))

        frame_number = int(filename.split('.')[-2])
        frame_idx = frame_number - 1 if self.dataset != 'shanghai' else frame_number - 1

        batch = []
        raw_frames = []  # store raw numpy frames for motion mask
        for i in range(self._num_frames):
            target_frame_path = self.videos[video_name]['frame'][frame_idx + i]
            
            image = np_load_frame(target_frame_path, self._resize_height, 
                                  self._resize_width, grayscale=True)
            
            if self.motion_mask:
                raw_frames.append(image)
            
            if self.transform is not None:
                batch.append(self.transform(image))

        # Extract Middle Label
        middle_offset = self._num_frames // 2
        label_idx = frame_idx + middle_offset
        
        if video_name in self.video_labels:
            safe_idx = min(label_idx, len(self.video_labels[video_name]) - 1)
            label = self.video_labels[video_name][safe_idx]
        else:
            label = 0

        img = OrderedDict()
        img['batch'] = np.stack(batch, axis=1)  # (C, T, H, W)
        img['label'] = label
        img['video_name'] = video_name
        img['index'] = label_idx * 200 // self.videos[video_name]['length']
        
        # Compute motion mask if enabled
        if self.motion_mask:
            mid_idx = self._num_frames // 2
            mask = compute_motion_mask(raw_frames, mid_idx,
                                       self.block_size, self.mask_ratio)
            img['motion_mask'] = mask  # (H, W) float32, 0=static 1=motion
        
        return img



class Reconstruction3DDataLoaderJump(Reconstruction3DDataLoader):
    def __getitem__(self, index):
        filename = os.path.basename(self.samples[index])

        parent_dir = os.path.dirname(self.samples[index])
        video_name = os.path.basename(parent_dir)

        frame_number_str = filename.split('.')[-2]

        if self.dataset == 'shanghai' and 'training' in self.samples[index]:
            frame_name = int(frame_number_str) - 1
        else:
            frame_name = int(frame_number_str)


        batch = []
        normal_batch = []
        jump = random.choice(self.jump)

        retry = 0
        while len(self.videos[video_name]['frame']) < frame_name + (self._num_frames-1) * jump and retry < 10:
            frame_name = np.random.randint(len(self.videos[video_name]['frame']))
            retry += 1

        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i*jump, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                  self._resize_width, grayscale=True)

            if self.transform is not None:
                batch.append(self.transform(image))

        if self.return_normal_seq:
            for i in range(self._num_frames):
                image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                      self._resize_width, grayscale=True)

                if self.transform is not None:
                    normal_batch.append(self.transform(image))
            return np.stack(batch, axis=1), np.stack(normal_batch, axis=1)

        else:
            return np.stack(batch, axis=1), normal_batch