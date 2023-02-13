import torch

torch.cuda.set_device(0)

import torch.nn as nn
import numpy as np
import argparse
import collections
import time
from sklearn.metrics import accuracy_score
import itertools
from tqdm import tqdm
import os
from Test_Loader import DataLoader
import cv2
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from dataloader.dataset import DesktopAssemblyDataset
from network import C3D_model
import glob

# DONE: Perform inference
# DONE: Use Queue based datastructure to keep track of frames
# DONE: Visualize the frames with action predictions

def Visualize_Labels(path_to_frames, path_to_labels, video_name, curr_preds):

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # label_dict ={
    #         "Pick up chip": 1,
    #         "Place chip on motherboard": 2,
    #         "Close cover": 3,
    #         "Pick up screw and screwdriver": 4,
    #         "Tighten screw": 5,
    #         "Plug stick in": 6,
    #         "Pick up fan": 7,
    #         "Place fan on motherboard": 8,
    #         "Tighten screw 1": 9,
    #         "Tighten screw 2": 10,
    #         "Tighten screw 3": 11,
    #         "Tighten screw 4": 12,
    #         "Put screwdriver down": 13,
    #         "Connect wire to motherboard": 14,
    #         "Pick up RAM": 15,
    #         "Install RAM": 16,
    #         "Pick up HDD": 17,
    #         "Install HDD": 18,
    #         "Connect wire 1 to HDD": 19,
    #         "Connect wire 2 to HDD": 20,
    #         "Pick up lid": 21,
    #         "Close lid": 22,
    #         "Background": 0
    #         }
    
    label_dict = {
        "Balling": 0,
        "Batting": 1,
        "Background": 2
    }

    reversed_label_dict = dict((reversed(item) for item in label_dict.items()))
    out = cv2.VideoWriter(os.path.join(path_to_labels, video_name+"_vis.avi"), fourcc, 10.0, (640,480))
    print(os.path.join(path_to_labels, video_name+"_vis.avi"))
    frames = sorted(glob.glob(os.path.join(path_to_frames, os.path.basename(video_name), "*")))    
    offset = 16
    #print(os.path.join(path_to_frames, os.path.basename(video_name)))
    print("Visualizing labels for: {}".format(video_name))
    for i, image in tqdm(enumerate(frames)):
       
        image = cv2.imread(image)
        if i >= 16:
            cv2.putText(image, str(reversed_label_dict[curr_preds[i - 16]]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 0, 255), 2)

        out.write(image)

    out.release()





def evaluate(net_path, batch_size, root, output_folder, force_rewrite = False):
    
    model = C3D_model.C3D(num_classes=23, pretrained=False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': 1e-3},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': 1e-3}]
  
    checkpoint = torch.load('{}'.format(net_path), map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    data_loader = DataLoader(video_folder = root, split = 'test')
    videos_info = data_loader.videos
  
    num_videos = len(videos_info.keys())
    scores_arr = []
    
    count = 0
    start_time = time.time()

    print("Number of vids:", num_videos)
    all_videos_scores_list = []
    all_videos_labels_list = []
    tot_inf_time = 0.0
    no_inf = 0

    for video_name, video in videos_info.items():
     
        count = count + 1
        
        if os.path.exists(os.path.join(output_folder,"{}.npy".format(video_name.replace(".mp4","")))) and not force_rewrite:
            continue

        length = video['length']
        num_his = 16
        scores_list =  []
        for i in tqdm(range(num_his, length, batch_size)):

            imgs = []
            
            labels = []
            cnt = 0
            while i + cnt  < length and cnt < batch_size:
     
                img, _ = data_loader.get_video_clips(video_name, i + cnt)
                
                imgs.append(img)
                cnt += 1

            if(len(imgs) >= 1):
                no_inf += 1  
                inf_start = time.time()

                imgs = torch.stack(imgs).float().to('cuda')
            
                outputs = model(imgs)
                inf_time = time.time() - inf_start

                tot_inf_time += inf_time
                
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                scores_list.extend(preds.cpu().numpy())

        np.save(os.path.join(output_folder,"{}.npy".format(video_name)), scores_list)
        Visualize_Labels(root, output_folder, video_name, scores_list)  
        all_videos_scores_list.extend(scores_list)

      
    print("Total Time elapsed: {}".format(time.time() - start_time))
    print("Total inference time: {}".format(tot_inf_time))
    print("Mean inference time: {}".format(tot_inf_time/no_inf))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default= 4, type=int, help='Batch size for training')
    parser.add_argument('--model_path', required = True, type = str, help = "Path to pretrained network")
    parser.add_argument('--output_path', required=True, type = str, help = "Path to store the output dictionaries and visualizations")
    parser.add_argument('--data_root', required=True, type=str, help = "Path to Test folder")
    parser.add_argument('--force_rewrite', action = "store_true", help = "Overwrite current predictions")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    evaluate(args.model_path, args.batch_size, args.data_root, args.output_path, force_rewrite= args.force_rewrite)
