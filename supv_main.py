#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import random
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser

from model_src.fully_main_model import supv_main_model as main_model

from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset
from losses import mmd_loss
import pdb

# =================================  seed config ============================
SEED = 456
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def get_flag_by_gt(is_event_scores):
    # is_event_scores: [bs, 10]
    scores_pos_ind = is_event_scores #> 0.5
    pred_temp = scores_pos_ind.long() # [B, 10]
    pred = pred_temp.unsqueeze(1) # [B, 1, 10]

    pos_flag = pred.repeat(1, 10, 1) # [B, 10, 10]
    pos_flag *= pred.permute(0, 2, 1)
    neg_flag = (1 - pred).repeat(1, 10, 1) # [B, 10, 10]
    neg_flag *= pred.permute(0, 2, 1)

    return pred_temp, pos_flag, neg_flag

def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_video_accuracy, best_accuracy_epoch = 0, 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')


    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data/', split='val'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''

    mainModel = main_model()

    num_params = count_parameters(mainModel)
    print("Total Parameter: \t%2.1fM" % num_params)
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    criterion = nn.BCELoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, epoch=0, eval_only=True)
        return

    # '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)


    '''Training and Testing'''
    for epoch in range(args.n_epoch):

        logger.info(f"\tnow epoch: {epoch}")

        loss = train_epoch(mainModel, train_dataloader, criterion, optimizer, a_mm=0)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, val_dataloader, criterion, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                )

        scheduler.step()

    print("-----------------------------")
    print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
    print("-----------------------------")


def train_epoch(model, train_dataloader, criterion, optimizer, a_mm=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feat_s4, audio_feature, labels, video_name = batch_data
        # visual_feat_s2, visual_feat_s3, visual_feat_s4, audio_feature, labels, video_name = batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()

        labels_foreground = labels[:, :, :-1]  # [bz, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)

        output, visual, audio = model(visual_feat_s4, audio_feature)

        output = output.view(output.size(0) * output.size(1), output.size(2))
        labels = labels.view(labels.size(0)*labels.size(1), labels.size(2))

        loss_mmd = mmd_loss(visual, audio, labels_BCE) * a_mm

        loss_event = criterion(output, labels)

        loss = loss_event + loss_mmd

        loss.backward()


        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feat_s4.size(0)*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()


        if n_iter % 100 == 0:
            print("loss", loss.item(), "loss_event", loss_event.item())

            for param_group in optimizer.param_groups:
                lr_ = param_group["lr"]
                logger.info(f"\t模型学习率为: {lr_:.6f}.")


    return losses.avg


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()


    total_acc = 0
    total_num = 0


    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feat_s4, audio_feature, labels, video_name = batch_data
        # visual_feat_s2, visual_feat_s3, visual_feat_s4, audio_feature, labels, video_name = batch_data
        # print(video_name)
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()

        bs = visual_feat_s4.size(0)
        output, visual, audio = model(visual_feat_s4, audio_feature) # [bz, 10, 29]

        labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))
        output = output.view(output.size(0) * output.size(1), output.size(2))
        total_acc += (output.squeeze(1).argmax(dim=-1) == labels.argmax(dim=-1)).sum()
        total_num += output.size(0)

    acc = (total_acc / total_num) * 100

    logger.info(
        f"\tEvaluation results (acc): {acc:.4f}%."
    )


    return acc


def save_checkpoint(state_dict, top1, task):
    model_name = f'{args.snapshot_pref}/model_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()


