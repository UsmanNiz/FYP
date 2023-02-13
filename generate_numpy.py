import os
import numpy as np


for file in os.listdir("c3d data"):
    if "txt" in file:
        preds = [] 
        with open("c3d data/"+file,'r') as fp:
            lines = fp.readlines()
            for line in lines:
                preds.append(int(line))
            
            preds = np.asarray(preds)
            data = file.replace("txt","npy")
            np.save("dataloader/cricket/train_labels/"+data,preds)

