import os
import glob
import torch
import numpy as np
from PIL import Image
import tqdm
from collections import OrderedDict
from time import time
import cv2
from collections import deque

#-------------------------------------------------------------------------------------------------------------------#

class DataLoader(object):
    def __init__(self,video_folder, split, annotations_path = None, resize_height = 128, resize_width = 171):

        self.split = split
        self.dir = video_folder
        self.videos = OrderedDict()
        self.gt = None
        self.crop_size = 112
        self.labels = None
        self.lab_dir = annotations_path    
        self.setup()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.frame_count = 8
        self.buffer_deque = deque()
        
    
    def __getitem__(self, video_name):
        assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        return self.videos[video_name]

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
  
        for i, video in enumerate(sorted(videos)):
            
            video_name = video.split('/')[-1]
            
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[video_name]['frame'].sort()
            
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            if self.split != "test":
            
                labels = np.load(os.path.join(self.lab_dir, '{}.npy'.format(video_name)))
                labels = labels[::3] #because of sub-sampling
                self.videos[video_name]['labels'] = labels

    def load_frames(self, file_dir, offset):
    
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        
        assert len(self.buffer_deque) == self.frame_count or len(self.buffer_deque) == 0, "Inconsistent length of frames"
        
        if len(self.buffer_deque) == self.frame_count:
            # pop last frame and add current frame
            self.buffer_deque.popleft()
            frame = cv2.imread(frames[offset])
            frame = np.array(cv2.resize(frame, (self._resize_width, self._resize_height))).astype(np.float64)
            self.buffer_deque.append(frame)
        elif len(self.buffer_deque) == 0:

            for i in range(offset - (self.frame_count - 1), offset + 1):
            
                frame = cv2.imread(frames[i])
                frame = np.array(cv2.resize(frame, (self._resize_width, self._resize_height))).astype(np.float64)
        
                self.buffer_deque.append(frame)
        
        return np.array(self.buffer_deque)
  

    def get_video_clips(self, video, start):

        if start == self.frame_count:
            self.buffer_deque = deque()
       
        buffer = self.load_frames(self.videos[video]["path"], start)

        buffer = self.normalize(buffer)
        buffer = self.crop(buffer, self.crop_size)
        buffer = self.to_tensor(buffer)
        
        if self.split != "test":
            
            label = self.videos[video]['labels'][start]
            return torch.from_numpy(buffer), torch.from_numpy(np.array(label))
        else:
            return torch.from_numpy(buffer), None
        
    def crop(self, buffer, crop_size):
        # randomly select time index for temporal jittering

        # Randomly select start indices in order to crop the video
     
        height_index = int((buffer.shape[1] - crop_size)/2)
        width_index =  int((buffer.shape[2] - crop_size)/2)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:,height_index:height_index + crop_size,
                    width_index:width_index + crop_size, :]

        return buffer
    def get_labels(self, video_name):
        return self.videos[video_name]['labels']
    

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

if __name__ == '__main__':
    data_loader = DataLoader('/home/sateesh/Desktop/Work/DesktopAssemblyBaseline/paritosh_codebase/data/test', '/home/sateesh/Desktop/Work/DesktopAssemblyBaseline/paritosh_codebase/data/test_labels', "test")
    videos_info = data_loader.videos
    num_videos = len(videos_info.keys())
    print(num_videos)
