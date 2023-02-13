import os
from sklearn.model_selection import train_test_split
import pickle as pkl
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import scipy.io

class DesktopAssemblyDataset(Dataset):


    def __init__(self, dataset = 'cricket' , split = 'train', num_classes = 9):

        self.root_dir = "dataloader/cricket/"
        folder = os.path.join(self.root_dir, split)
        folder_labels = os.path.join(self.root_dir, split+"_labels")
        self.split = split
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        self.num_classes = num_classes
        self.current_file_index = 0
        
        self.fnames, self.f_lengths, self.labels = [], [], []
        for fname in sorted(os.listdir(folder)):

            self.fnames.append(os.path.join(folder, fname))
            self.f_lengths.append(len(os.listdir(self.fnames[-1])))
            self.labels.append(os.path.join(folder_labels, fname+".npy"))
                    
        
            
    def __getitem__(self, index):
        
        
        frame_index = np.random.randint(16, self.f_lengths[index])

        buffer = self.load_frames(self.fnames[index], frame_index)

       
        labels = np.load(self.labels[index])
        # labels = labels[::3] # subsampling because of lower framerate, Labels are at 30 fps and the dataset is at 10 fps
        label  = labels[frame_index]
        
        buffer = self.normalize(buffer)
        buffer = self.crop(buffer, self.crop_size)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(np.array(label))

    def __len__(self):
        
        return len(self.fnames)
    


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir, offset):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        
        frame_count = 16 #fixed for c3d
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        
        for i in range(offset - (frame_count - 1), offset + 1):
            frame = cv2.imread(frames[i])
            
            frame = np.array(cv2.resize(frame, (self.resize_width, self.resize_height))).astype(np.float64)
            
            buffer[i - offset + (frame_count - 1)] = frame

        return buffer

    def crop(self, buffer, crop_size):
        # randomly select time index for temporal jittering

        # Randomly select start indices in order to crop the video

        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:,height_index:height_index + crop_size,
                    width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    train_data = DesktopAssemblyDataset(dataset='CAE', split='test')
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, worker_init_fn = lambda _: np.random.seed() )
    #print(train_loader.next())
    
    for i in range(20):
      
        for i, sample in enumerate(train_loader):
            inputs = sample[0]
            labels = sample[1]
            print(inputs.shape)
            print(labels)
