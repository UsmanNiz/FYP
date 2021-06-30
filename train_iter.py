import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from dataloaders.aug_slowFast_dataset import DesktopAssemblyDataset
from network import C3D_model_8 as C3D_model
import argparse
import math
# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
plot_every = 25
print_every = 5
train_batch_size = 8
val_batch_size = 5


save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def train_model(dataset, save_dir_root, lr,
                max_iters, resume_iter, save_epoch, test_interval, run_id, num_classes=23, useVal=True):
    
    best_val_loss = 100


    save_dir = os.path.join(save_dir_root, 'run', 'run_' + run_id)
    modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
    saveName = modelName + '-' + dataset

    
    last_change = 0
    #Initializing model
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    
    optimizer = optim.Adam(train_params, lr=lr)
    #loading Pre-Trained model
    
    if resume_iter == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_iter-' + str(resume_iter - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU

        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_iter-' + str(resume_iter - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)
   
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on with Split: '.format(split))
    #Initialize data loaders
    train_dataloader = DataLoader(DesktopAssemblyDataset(dataset=dataset, split=split), batch_size = train_batch_size, shuffle=True, num_workers=4, worker_init_fn = worker_init_fn)
    val_dataloader   = DataLoader(DesktopAssemblyDataset(dataset=dataset, split='val'), batch_size= val_batch_size, num_workers=4, worker_init_fn = worker_init_fn)
    trainval_loaders = {'train': train_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train']}
    val_size = len(val_dataloader.dataset)

    itr = resume_iter
    epoch = int(resume_iter/(math.ceil(trainval_sizes['train']/train_batch_size)))
    running_loss = 0.0
    running_corrects = 0.0
    start_time = timeit.default_timer()
    quit_training = False
    
    while itr < max_iters and not quit_training:
        
        
        np.random.seed(55+epoch)
        if (epoch % 25 == 0): 
            print("Epoch: {}, iter: {}".format(epoch, itr))
       
        for phase in ['train']:
            
            # reset the running loss and corrects
            

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                
                model.train()
            else:
                model.eval()

            for inputs, labels in (trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                
                labels = Variable(labels).to(device).long()
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                # print(outputs.shape)
                # print('\n\n\n\n\n\n\n\n\n\n\n')
                
                loss = criterion(outputs, labels)
                itr += 1
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if(itr % plot_every == 0):
                    
                    itr_loss = running_loss / (plot_every *  train_batch_size)
                    itr_acc = running_corrects.double() / (plot_every * train_batch_size)

                    writer.add_scalar('data/train_loss', itr_loss, itr)
                    writer.add_scalar('data/train_acc', itr_acc, itr)

                    running_loss = 0
                    running_corrects = 0

                    print("[{}] Iter: {}/{} Loss: {} Acc: {}".format(phase, itr, max_iters, itr_loss, itr_acc))
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                    start_time = timeit.default_timer()
            
                if itr % save_epoch == 0:
                    torch.save({
                        'epoch': itr,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                    }, os.path.join(save_dir, 'models', saveName + '_iter-' + str(itr) + '.pth.tar'))
                    print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_iter-' + str(itr) + '.pth.tar')))

                if useVal and itr % test_interval == 0:
                    model.eval()
                    start_time = timeit.default_timer()

                    running_loss_val = 0.0
                    running_corrects_val = 0.0
                    # run 50 steps on validation set
                    for i in range((val_iters)):
                        np.random.seed(55+i)
                        for inputs, labels in tqdm(val_dataloader):
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            with torch.no_grad():
                                outputs = model(inputs)
                            
                            probs = nn.Softmax(dim=1)(outputs)
                            preds = torch.max(probs, 1)[1]
                            
                            loss = criterion(outputs, labels)

                            running_loss_val += loss.item() * inputs.size(0)
                            running_corrects_val += torch.sum(preds == labels.data)

                    epoch_loss = running_loss_val / (val_size * val_iters)
                    epoch_acc = running_corrects_val.double() / (val_size * val_iters)

                    writer.add_scalar('data/val_loss_epoch', epoch_loss, itr)
                    writer.add_scalar('data/val_acc_epoch', epoch_acc, itr)

                    print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(itr+1, max_iters, epoch_loss, epoch_acc))
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                    if epoch_loss < best_val_loss:

                        best_val_loss = epoch_loss
                        best_val_acc = epoch_acc
                        last_change = 0
                        torch.save({
                        'epoch': itr,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                    }, os.path.join(save_dir, 'models', saveName + '_iter-' + str(itr) + '.pth.tar'))
                        print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_iter-' + str(itr) + '.pth.tar')))


                    
                    else:
                        last_change += 1
                        print("Training loss decrease count: {}".format(last_change))
                    
                    if last_change >= 10:
                        quit_training = True
                    
        epoch += 1
       


    writer.close()
def worker_init_fn(worker_id):

    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DesktopAssembly", type=str, help='Dataset to run experiment on')
    parser.add_argument('--resume_itr', default=0, required= False , type=int, help="Iteration to start training from")
    parser.add_argument('--nTestInterval', default=800, required= False , type=int, help="Number of iterations before testing the model")
    parser.add_argument('--snapshot', default=4000, required= False , type=int, help="Number of iterations before saving the model")
    parser.add_argument('--learningrate', default=1e-4, required= False, type=float)
    parser.add_argument('--val_iters', default=40, required= False, type=int, help="Number of iterations to run the validation set for")
    parser.add_argument('--split', default="train", required= False, type=str, help="Split to use for training")
    parser.add_argument('--max_iters', default=100000, required= False, type=int, help="Maximum number of iterations to run the model")
    parser.add_argument('--run_id',  required= True, type=str, help="Experiment name to save model")
    
    
    args = parser.parse_args()

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    max_iters = args.max_iters
    resume_iter = args.resume_itr
    useVal = True # See evolution of the test set when training
    nTestInterval = args.nTestInterval # Run on test set every nTestInterval epochs
    snapshot = args.snapshot # Store a model every snapshot epochs
    lr = args.learningrate # Learning rate
    val_iters = args.val_iters
    split = args.split
    run_id = args.run_id
    dataset = args.dataset



    train_model(dataset=dataset, save_dir_root=save_dir_root, lr=lr,
                max_iters=max_iters, resume_iter = resume_iter, save_epoch=snapshot, useVal=useVal, test_interval=nTestInterval, run_id = run_id)
