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

from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from dataloaders.dataset import DesktopAssemblyDataset
from network import C3D_model, R2Plus1D_model, R3D_model


def evaluate(net_path, batch_size, root, annotations_path, output_folder, force_rewrite = False):
    
    model = C3D_model.C3D(num_classes=23, pretrained=False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': 1e-3},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': 1e-3}]
  
    checkpoint = torch.load('{}'.format(net_path), map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    
    data_loader = DataLoader(video_folder = root, annotations_path = annotations_path, split = 'val')
    videos_info = data_loader.videos
  
    num_videos = len(videos_info.keys())
    scores_arr = []
    y_true_arr = []

    count = 0
    start_time = time.time()

    print("Number of vids:", num_videos)
    print("HERE") 
    all_videos_scores_list = []
    all_videos_labels_list = []
    tot_inf_time = 0.0
    no_inf = 0

    for video_name, video in videos_info.items():

        
        print(video_name)
        count = count + 1
        
        if os.path.exists(os.path.join(output_folder,"{}.npy".format(video_name))) and not force_rewrite:
            continue

        length = video['length']
        num_his = 16
        scores_list =  []
        true_scores_list = []
        
        for i in tqdm(range(num_his, length, batch_size)):
            
                    
            imgs = []
            
            labels = []
            cnt = 0
            while i + cnt  < length and cnt < batch_size:
     
                img, label = data_loader.get_video_clips(video_name, i + cnt)
                
                imgs.append(img)
                labels.append(label)
               
                
                true_scores_list.extend([label.numpy()])
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

        
        print("Length of scores {}, length of labels {}".format(len(scores_list), len(true_scores_list)))

        all_videos_scores_list.extend(scores_list)
        all_videos_labels_list.extend(true_scores_list)

        print("length of overall scores {}, length of overall labels {}".format(len(all_videos_scores_list), len(all_videos_labels_list)))
    
    print(accuracy_score(all_videos_labels_list, all_videos_scores_list))
    print("Total Time elapsed: {}".format(time.time() - start_time))
    print("Total inference time: {}".format(tot_inf_time))
    print("Mean inference time: {}".format(tot_inf_time/no_inf))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default= 4, type=int, help='Batch size for training')
    parser.add_argument('--model_path', required = True, type = str, help = "Path to pretrained network")
    parser.add_argument('--output_path', required=True, type = str, help = "Path to store the output dictionaries")
    parser.add_argument('--data_root', required=True, type=str, help = "Path to Test folder")
    parser.add_argument('--annotations_path', required= True , type=str, help = "Path to label folder")
    parser.add_argument('--force_rewrite', action="store_true", help = "Overwrite current predictions")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    evaluate(args.model_path, args.batch_size, args.data_root, args.annotations_path, args.output_path, force_rewrite = args.force_rewrite)
