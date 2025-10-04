import os
import h5py
import torch
import pandas as pd
import random
import numpy as np
import math
import glob
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()

        self.split = split

        self.cnn14_feature_path = "/data01/home/zhangpufen/AVE_data/object_feature_attr/cnn14_feat/"

        self.swin_s4feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat/"
        self.swin_s2feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat_s2/"
        self.swin_s3feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat_s3/"

        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')

        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5')

        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.raw_gt = pd.read_csv(data_root+"/Annotations.txt", sep="&", header=None)
        self.h5_isOpen = False


    def __getitem__(self, index):

        if not self.h5_isOpen:

            self.labels = h5py.File(self.labels_path, 'r')['avadataset']


            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True

        sample_index = self.sample_order[index]

        file_name = self.raw_gt.iloc[sample_index][1]

        swin_s4visual_feat_path = self.swin_s4feature_path + '/' + file_name + '.npy'
        swin_s4visual_feat = np.load(swin_s4visual_feat_path)
        visual_feat = np.array(swin_s4visual_feat, dtype=np.float64)

        # swin_s2visual_feat_path = self.swin_s2feature_path + '/' + file_name + '.npy'
        # swin_s2visual_feat = np.load(swin_s2visual_feat_path)
        # visual_feat = np.array(swin_s2visual_feat, dtype=np.float64)

        # swin_s3visual_feat_path = self.swin_s3feature_path + '/' + file_name + '.npy'
        # swin_s3visual_feat = np.load(swin_s3visual_feat_path)
        # visual_feat = np.array(swin_s3visual_feat, dtype=np.float64)

        audio_feat_path = self.cnn14_feature_path + file_name + '.npy'
        audio_feat = np.load(audio_feat_path)
        audio_feat = np.array(audio_feat, dtype=np.float64)

        label = self.labels[sample_index]

        return visual_feat, audio_feat, label, file_name


    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num


if __name__ == '__main__':

    train_dataloader = DataLoader(
            AVEDataset('/mnt/d/AVE_data/', split='train'),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        )

    for n_iter, batch_data in enumerate(train_dataloader):
        if n_iter==1:
            break
        visual_feat, box_feat, audio_feature, labels, graph = batch_data
        print(graph.shape)


        # print(graph)
    print("完成")
        # print(visual_feature.shape)


