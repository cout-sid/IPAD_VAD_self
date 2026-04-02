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


class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=16,
                 img_extension='.jpg', dataset='ped2', jump=[2], hold=[2], return_normal_seq=False):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames

        self.extension = img_extension
        self.dataset = dataset

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
        # index = 8
        # video_name = self.samples[index].split('\\')[-2]
        # if self.dataset == 'shanghai' and 'training' in self.samples[index]:
        #     frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2]) - 1
        # else:
        #     frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])

        # video_name = os.path.basename(os.path.normpath(self.samples[index]))
        filename = os.path.basename(self.samples[index])

        parent_dir = os.path.dirname(self.samples[index])
        video_name = os.path.basename(parent_dir)

        # Split by '.' and get the number part (e.g., '001')
        # This works for '001.jpg' -> ['001', 'jpg'] -> '001'
        frame_number_str = filename.split('.')[-2]

        if self.dataset == 'shanghai' and 'training' in self.samples[index]:
            frame_name = int(frame_number_str) - 1
        else:
            frame_name = int(frame_number_str)

        batch = []
        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width, grayscale=False)
# np_load_frame returns shape => (h,w,3)
            
            if self.transform is not None:
                batch.append(self.transform(image))
# no automatic scaling normally we scale (0,255)=>(0.0,1.0) but 
# np_load_frame scaled it already to (-1.0,1.0) which is float so transform doesn't scale it now

        # batch:len=16 ,batch[0]:torch(3,256,256)
        img = OrderedDict()
        img['batch'] = np.stack(batch, axis=1)
        img['index'] = frame_name*200//len(self.videos[video_name]['frame'])
        # return np.stack(batch, axis=1)
        return img

    def __len__(self):
        return len(self.samples)
    
class TestDataLoader(Reconstruction3DDataLoader):
    def __init__(self, video_folder, label_folder, transform, resize_height, resize_width, 
                 num_frames=16, img_extension='.jpg', dataset='ped2'):
        # 1. Initialize the parent class to set up video paths and frame lists
        super(TestDataLoader, self).__init__(video_folder, transform, resize_height, resize_width, 
                                             num_frames, img_extension, dataset)
        
        self.label_dir = label_folder
        self.video_labels = {}
        
        # 2. Load labels and sync with existing video lengths
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

                # Sync: Use the smaller of the two to ensure every frame has a label
                min_len = min(num_frames, num_labels)
                self.video_labels[video_name] = labels[:min_len]
                
                # Update parent dictionary if we had to trim frames to match labels
                if num_frames > min_len:
                    self.videos[video_name]['frame'] = self.videos[video_name]['frame'][:min_len]
                    self.videos[video_name]['length'] = min_len
            else:
                print(f"Warning: Label file {label_path} not found for video {video_name}")

    def __getitem__(self, index):
        # Path of the first frame in the sequence
        full_path = self.samples[index]
        filename = os.path.basename(full_path)
        video_name = os.path.basename(os.path.dirname(full_path))

        # Determine frame index from filename (e.g., '081.jpg' -> 81)
        # Note: If filenames start at 001, frame_idx 0 corresponds to '001.jpg'
        frame_number = int(filename.split('.')[-2])
        frame_idx = frame_number - 1 if self.dataset != 'shanghai' else frame_number - 1

        # 1. Load the 16-frame clip
        batch = []
        for i in range(self._num_frames):
            # Target is current frame + step
            # parent's get_all_samples ensures this index is always valid
            target_frame_path = self.videos[video_name]['frame'][frame_idx + i]
            
            image = np_load_frame(target_frame_path, self._resize_height, 
                                  self._resize_width, grayscale=True)
            
            if self.transform is not None:
                batch.append(self.transform(image))

        # 2. Extract Middle Label
        # For num_frames=16, the middle frame is index 8 (the 9th frame)
        middle_offset = self._num_frames // 2
        label_idx = frame_idx + middle_offset
        
        # Pull the specific label for this middle frame
        if video_name in self.video_labels:
            # clip index just in case of very short video/label mismatch
            safe_idx = min(label_idx, len(self.video_labels[video_name]) - 1)
            label = self.video_labels[video_name][safe_idx]
        else:
            label = 0

        # 3. Format output
        img = OrderedDict()
        img['batch'] = np.stack(batch, axis=1) # Shape: (C, T, H, W)
        img['label'] = label
        img['video_name'] = video_name
        img['index'] = label_idx * 200 // self.videos[video_name]['length']
        
        return img



class Reconstruction3DDataLoaderJump(Reconstruction3DDataLoader):
    def __getitem__(self, index):
        # index = 8
        # video_name = self.samples[index].split('\\')[-2]
        # if self.dataset == 'shanghai' and 'training' in self.samples[index]:  # bcos my shanghai's start from 1
        #     frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2]) - 1
        # else:
        #     frame_name = int(self.samples[index].split('\\')[-1].split('.')[-2])

        # video_name = os.path.basename(os.path.normpath(self.samples[index]))
        filename = os.path.basename(self.samples[index])

        parent_dir = os.path.dirname(self.samples[index])
        video_name = os.path.basename(parent_dir)

        # Split by '.' and get the number part (e.g., '001')
        # This works for '001.jpg' -> ['001', 'jpg'] -> '001'
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
            # reselect the frame_name
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


