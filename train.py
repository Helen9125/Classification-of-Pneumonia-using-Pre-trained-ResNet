#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image

#import seaborn as sns
from matplotlib.ticker import MaxNLocator

import pandas as pd


# In[ ]:


torch.set_default_device('cuda:2')


# In[ ]:


def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn


# In[ ]:


def plot_accuracy(train_acc_list, val_acc_list):
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, linewidth=2.0, color='royalblue', label='Training accuracy')
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, linewidth=2.0, color='orange', label='Validation accuracy')

    epochs = [1, 10, 20, 30, 40, 50]
#epochs = [1, 5, 10, 15, 20]
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy Over Epochs', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    print('saving acc picture')
    plt.savefig('./result/resnet18/acc_64_2.jpg')


# In[ ]:


def plot_f1_score(f1_score_list, val_f1_score_list):
    # TODO plot testing f1 score curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(f1_score_list) + 1), f1_score_list, linewidth=2.0, color='royalblue', label='Training f1_score')
    plt.plot(range(1, len(val_f1_score_list) + 1), val_f1_score_list, linewidth=2.0, color='orange', label='Validation f1_score')
    
    epochs = [1, 10, 20, 30, 40, 50]
    
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('f1_score', fontsize=14)
    plt.title('f1_score Over Epochs', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    print('saving f1_score picture')
    plt.savefig('./result/resnet18/f1_score_64_2.jpg')
    pass


# In[ ]:


def plot_confusion_matrix(confusion_matrix):
    # TODO plot confusion matrix
    cm_array = np.array(best_c_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_array, display_labels = [False, True])

    cm_display.plot()
    plt.savefig('./result/resnet18/confusion_matrix_64_2.jpg')
    plt.show()
    



# In[ ]:


class ChestXrayDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.loaded_imgs = self.load_img()

        print('loading data')
        

    def load_img(self):
        imgs = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_path, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                imgs.append((img_path, self.class_to_idx[cls]))
        return imgs

    def __len__(self):
        return len(self.loaded_imgs)

    def __getitem__(self, idx):
        imgs = self.load_img()
        img_path, label = imgs[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)        
        return img, label



# In[ ]:


def train(device, train_loader, val_loader, model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    val_f1_score_list = []
    best_c_matrix = []
    #val_f1_score_list = []
    best_loss = float('inf')

    print('start training')
    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            #for _, data in enumerate(tqdm(train_loader)):
            for inputs, labels in train_loader:
                #inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            f1_score = (2*tp) / (2*tp+fp+fn)
            
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')
            print(f'↳ Training f1_score: {f1_score}%')

        # write validation if you needed
        # val_acc, f1_score, c_matrix = test(val_loader, model)

        train_acc_list.append(train_acc)
        f1_score_list.append(f1_score)
        #f1_score_list.append(f1_score)

        if True:
            #val_acc, f1_score, c_matrix = test(val_loader, model)
            model.eval()
            
            val_avg_loss = 0.0
            val_acc = 0.0
            val_tp, val_tn, val_fp, val_fn = 0, 0, 0, 0     
            
            with torch.no_grad():
                #for _, val_data in enumerate(tqdm(val_loader)):
                for val_inputs, val_labels in val_loader:
                    #val_inputs, val_labels = val_data
                    val_inputs = inputs.to(device)
                    val_labels = labels.to(device)

                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)


                    val_avg_loss += val_loss.item()
                    val_outputs = torch.max(val_outputs, 1).indices
                    val_sub_tp, val_sub_tn, val_sub_fp, val_sub_fn = measurement(val_outputs, val_labels)
                    val_tp += val_sub_tp
                    val_tn += val_sub_tn
                    val_fp += val_sub_fp
                    val_fn += val_sub_fn       

                val_avg_loss /= len(train_loader.dataset)
                val_acc = (val_tp+val_tn) / (val_tp+val_tn+val_fp+val_fn) * 100
                val_f1_score = (2*val_tp) / (2*val_tp+val_fp+val_fn)
                print(f'↳ val_Loss: {val_avg_loss}')
                print(f'↳ Valid Acc.(%): {val_acc:.2f}%')
                print(f'↳ Valid f1_score: {val_f1_score}%')
                val_acc_list.append(val_acc)
                val_f1_score_list.append(val_f1_score)
                #best_c_matrix.append(c_matrix.item())

                good_model = val_avg_loss < best_loss
                if good_model:
                    best_loss = val_avg_loss
                    print(f'Saving at epoch {epoch + 1}')
                    torch.save(model.state_dict(), 'model_weights_resnet18_64_2.pt')
            model.train()

    

    return train_acc_list, val_acc_list, f1_score_list,  val_f1_score_list


# In[ ]:


def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix


# In[ ]:


parser = ArgumentParser(allow_abbrev=True)


# In[ ]:


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='./data/chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')
    







    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree), #, resample=False),
                                                                transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2), shear = 10),
                                                                transforms.ColorJitter(0.2, 0.2, 0.2, 0),
                                                                transforms.ToTensor()]))
    val_dataset = ImageFolder(root=os.path.join(args.dataset, 'val'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree), #, resample=False),,
                                                                transforms.ToTensor()]))
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=torch.Generator(device=device))
    
    # define model
    model = models.resnet18(pretrained=True)
    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training
    train_acc_list, val_acc_list, f1_score_list, val_f1_score_list = train(device, train_loader, val_loader, model, criterion, optimizer)
    # testing
    test_acc, f1_score, best_c_matrix = test(test_loader, model)
    print('test_acc:', test_acc, 'f1_score:', f1_score, 'best_c_matrix:', best_c_matrix)

    # plot
    plot_accuracy(train_acc_list, val_acc_list)
    plot_f1_score(f1_score_list, val_f1_score_list)
    plot_confusion_matrix(best_c_matrix)

    acc_df = pd.DataFrame({'train acc': train_acc_list, 'valid acc': val_acc_list})
    f1_score_df = pd.DataFrame({'train f1_score': f1_score_list, 'valid f1_score': val_f1_score_list})

    print('saving to csv')
    acc_df.to_csv('./result/resnet18/c_acc_64_2.csv')
    f1_score_df.to_csv('./result/resnet18/c_f1_score_64_2.csv')


# In[ ]:




