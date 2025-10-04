import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split


        self.noisy_audio_feature_path = os.path.join(data_root, 'audio_feature_noisy.h5')   # only background

        self.cnn14_feature_path = "/data01/home/zhangpufen/AVE_data/object_feature_attr/cnn14_feat/"

        # s4 feature
        self.swin_feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat/"

        # self.swin_feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat_s2/"

        # self.swin_feature_path = "/data01/home/zhangpufen/AVE_data/swinv2_large_feat_s3/"

        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5') # original labels for testing

        self.bg_raw_gt = pd.read_csv(data_root + "/modified_noisydataset.txt", sep="&", header=None)

        self.raw_gt = pd.read_csv(data_root + "/Annotations.txt", sep="&", header=None)

        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # video-level labels

        self.dir_labels_bg_path = os.path.join(data_root, 'labels_noisy.h5')  # only background

        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')

        self.h5_isOpen = False


    def __getitem__(self, index):

        if not self.h5_isOpen:

            self.sample_order = h5py.File(self.sample_order_path, 'r')['order'] # 所有的训练样本


            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']

            if self.split == 'train':
                self.negative_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
                self.negative_audio_feature = '/data01/home/zhangpufen/AVE_data/cnn14_nosiy_feat_16k/'

            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']

            self.h5_isOpen = True

        clean_length = len(self.sample_order)
        # print(clean_length) # 3339,训练集的大小

        if index >= clean_length:
            # file_name = self.raw_gt.iloc[self.sample_order[index]][1]
            # print(file_name) #  Index (3441) out of range for (0-3338)
            # print(index)
            valid_index = index - clean_length
            # print("valid_index", valid_index)

            file_name = self.bg_raw_gt.iloc[valid_index][1]

            # neg_visual_feat_path = self.negative_visual_feature + neg_file_name + '.npy'
            neg_visual_feat_path = self.swin_feature_path + file_name + '.npy'

            visual_feat = np.load(neg_visual_feat_path)
            visual_feat = np.array(visual_feat, dtype=np.float64)
            # print(visual_feat.shape) # (10, 7, 7, 512)

            neg_audio_feat_path = self.negative_audio_feature + file_name + '.npy'
            audio_feat = np.load(neg_audio_feat_path)
            audio_feat = np.array(audio_feat, dtype=np.float64)

            # audio_feat = self.noisy_audio_feature[valid_index]

            label = self.negative_labels[valid_index] # shape=[29,]
            # print(label)
        else:
            # test phase or negative training samples
            sample_index = self.sample_order[index]
            file_name = self.raw_gt.iloc[sample_index][1]
            # visual_feat_path = self.visual_feature_path + file_name + '.npy'
            visual_feat_path = self.swin_feature_path + file_name + '.npy'
            visual_feat = np.load(visual_feat_path)
            visual_feat = np.array(visual_feat, dtype=np.float64)

            audio_feat_path = self.cnn14_feature_path + file_name + '.npy'
            audio_feat = np.load(audio_feat_path)
            audio_feat = np.array(audio_feat, dtype=np.float64)

            # audio_feat = self.audio_feature[sample_index]

            if self.split == 'train':
                label = self.clean_labels[sample_index] # [29,]
                # print(label)
            else:
                # for testing
                label = self.labels[sample_index]

        return visual_feat, audio_feat, label, file_name


    def __len__(self):
        if self.split == 'train':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            noisy_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']
            # print(len(noisy_labels)) # 178
            length = len(sample_order) + len(noisy_labels)
        elif self.split == 'test':
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            length = len(sample_order)
        else:
            raise NotImplementedError

        return length


if __name__ == '__main__':
    train_dataloader = DataLoader(
            AVEDataset('/data01/home/zhangpufen/AVE_data', split='train'),
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=False)

    for n_iter, batch_data in enumerate(train_dataloader):
        '''Feed input to model'''
        visual_feature, audio_feature, labels, video_name = batch_data

        print(labels)

        # print(visual_feature.shape)


