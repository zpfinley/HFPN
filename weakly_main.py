import os
import time
import random
import json
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from configs.opts import parser
from model_src.weak_main_model import weak_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset_weak import AVEDataset

# =================================  seed config ============================

SEED = 666
# SEED = 456
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


def main(filter_num=0):
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving model '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval.log.')

    '''Dataset'''

    train_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )

    val_dataloader = DataLoader(
        AVEDataset('/data01/home/zhangpufen/AVE_data', split='val'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )

    '''model setting'''
    mainModel = main_model()

    num_params = count_parameters(mainModel)
    print("Total Parameter: \t%2.1fM" % num_params)

    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=40)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion_event = nn.CrossEntropyLoss().cuda()

    criterion_event = nn.MultiLabelSoftMarginLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                )
        scheduler.step()


def train_epoch(model, train_dataloader, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels, video_name = batch_data
        # s2, s3, s4, audio_feature, labels, video_name = batch_data
        # visual_feature = s4
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.cuda()
        scores, _ = model(visual_feature, audio_feature)
        # scores, _ = model(s2, s3, s4, audio_feature)

        loss_event_class = criterion_event(scores, labels.double())

        loss = loss_event_class
        loss.backward()

        '''Compute Accuracy'''
        acc = torch.tensor([0])
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
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
        visual_feature, audio_feature, labels, video_name = batch_data
        # s2, s3, s4, audio_feature, labels, video_name = batch_data

        labels = labels.cuda()
        # labels = labels.numpy()
        # bs = visual_feature.size(0)
        _, event_scores = model(visual_feature, audio_feature)
        # _, event_scores = model(s2, s3, s4, audio_feature)

        labels = labels.view(labels.size(0) * labels.size(1), labels.size(2))
        output = event_scores.view(event_scores.size(0) * event_scores.size(1), event_scores.size(2))
        total_acc += (output.squeeze(1).argmax(dim=-1) == labels.argmax(dim=-1)).sum()
        total_num += output.size(0)

    acc = (total_acc / total_num) * 100

    logger.info(
        f"\tEvaluation results (acc): {acc:.6f}%."
    )

    return acc


def save_checkpoint(state_dict, top1, task):
    model_name = f'{args.snapshot_pref}/model_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':

    main()
