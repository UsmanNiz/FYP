import numpy as np
import torch

def crop(buffer, crop_size):
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

def normalize(buffer):
    for i, frame in enumerate(buffer):
        frame -= np.array([[[90.0, 98.0, 102.0]]])
        buffer[i] = frame

    return buffer

def to_tensor(buffer):
    return buffer.transpose((3, 0, 1, 2))