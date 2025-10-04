import json
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# def generate_category_list():
#     file_path = 'please change this to the path of VggsoundAVEL100kCategories.txt'
#     category_list = []
#     with open(file_path, 'r') as fr:
#         for line in fr.readlines():
#             category_list.append(line.strip())
#     return category_list

def load_anno(path):
    with open(path, 'r') as f:
        anno = json.load(f)
    f.close()
    return anno

def load_category(path):
    with open(path, 'r') as f:
        category_name = json.load(f)
    f.close()
    category_name_list = category_name.keys()
    class_label = range(0, len(category_name_list))
    category_label = dict(zip(category_name_list, class_label))
    # print(category_label)
    return category_label

class VGGSoundAVELDatasetFully(Dataset):
    # for AVEL task
    def __init__(self, split='train'):
        super(VGGSoundAVELDatasetFully, self).__init__()

        self.meta_csv_path = '/data01/home/zhangpufen/AVEL_100K_dataset/vggsound-avel100k.csv'
        self.avc_label_base_path = '/data01/home/zhangpufen/AVEL_100K_dataset/vggsound-avel100k_annos.json'
        self.category_name_path = '/data01/home/zhangpufen/AVEL_100K_dataset/category_labels.json'

        self.audio_feat_path = '/data01/home/zhangpufen/AVEL_100K_dataset/VGG_AVEL100K_cnn14_16k_feat/'
        # self.swin_feat_path = '/data01/home/zhangpufen/AVEL_100K_dataset/swinv2_large_feat/'
        # self.swin_feat_path = '/data01/home/zhangpufen/AVEL_100K_dataset/swinv2_large_feat_s2/'
        self.swin_feat_path = '/data01/home/zhangpufen/AVEL_100K_dataset/swinv2_large_feat_s3/'


        self.avc_label = load_anno(self.avc_label_base_path)
        self.category_dirt = load_category(self.category_name_path)
        # print(len(self.category_dirt)) # 141
        all_df = pd.read_csv(self.meta_csv_path)
        self.split_df = all_df[all_df['split'] == split]
        print(f'{len(self.split_df)}/{len(all_df)} videos are used for {split}')

    def _obtain_avel_label(self, avc_label, class_id):
        # avc_label: [1, 10]
        T, category_num = 10, len(self.category_dirt)
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        avc_label = np.array(avc_label)
        bg_flag = 1 - avc_label
        label[:, class_id] = avc_label
        label[:, -1] = bg_flag

        return label

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, video_id = one_video_df['category'], one_video_df['video_id']
        event_label = self.avc_label[video_id]['label']
        class_label = self.category_dirt[category]
        audio_fea = np.load(self.audio_feat_path + video_id + '.npy')

        ave_labels = self._obtain_avel_label(event_label, class_label)

        if audio_fea.shape[0] < 10:
            audio_pad_len = 10 - audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (audio_pad_len, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        # 大于十秒取前十秒
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        # total_img = np.load(self.video_feat_path + video_id + '.npy')
        total_img = np.load(self.swin_feat_path + video_id + '.npy')
        total_img = torch.from_numpy(total_img)

        if total_img.size(0) < 10:
            pad_len = 10 - total_img.size(0)
            add_arr = total_img[-1, :, :].repeat(pad_len, 1).view(pad_len, total_img.size(1), total_img.size(2))
            total_img = torch.cat([total_img, add_arr], dim=0)

        elif total_img.shape[0] > 10:
            total_img = total_img[:10, :, :]
        ave_labels = torch.from_numpy(ave_labels)

        audio_fea = torch.from_numpy(audio_fea)

        return total_img, audio_fea, ave_labels, video_id


    def __len__(self,):
        return len(self.split_df)


if __name__ == '__main__':

    train_dataloader = DataLoader(
        VGGSoundAVELDatasetFully(split='train'),
        batch_size=128,
        shuffle=False,
        num_workers=16,
        pin_memory=False
    )

    for n_iter, batch_data in enumerate(train_dataloader):
        # if n_iter==1:
        #     break
        labels, total_img, audio_fea, video_id = batch_data
        # print(total_img.shape)
        print(video_id)

        """some processing on the labels"""
        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1) # [32, 10], [32, 10]
        labels_event, _ = labels_evn.max(-1) # [32]
        # print(labels_BCE)
        # print(labels_event)
        # print(labels)
    print("**********************finish**************************")






