import sys
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import simsiam.loader
import simsiam.builder
from resnet_cifar import BasicBlock as BasicBlockCifar
from resnet_cifar import ResNet as ResNetCifar

from fractal_datamodule.selfsup_fractal_datamodule import SelfSupMultiFractalDataModule
from fractal_datamodule.datasets.selfsup_fractaldata import SelfSupGenerator
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union


class MyImageFolder(datasets.ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            x1, x2 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return x1, x2, target


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ('mobilenetv2', 'swintiny')


def parse_multifractal_setting(setting_str):
    tokens = setting_str.split('_')
    assert(tokens[0] == 'multifractal')
    assert(tokens[1] in ['224'])

    fractal_args = {'imgsize': int(tokens[1])}
    for token in tokens[2:]:
        k, v = token.split('-')
        if k in ['cls', 'sys', 'mnobjs', 'genaugs']:
            fractal_args[k] = int(v)
        elif k == 'colormode':
            fractal_args[k] = v
        elif k in ['w', 'wo']:
            fractal_args[v] = (k == 'w')
        else:
            print('Unrecognized multifractal settings', k, v)
            raise NotImplementedError()
    return fractal_args


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 512), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume-only-weights', action='store_true', default=False,
                        help='When resuming a model, only resume its weights')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # simsiam specific configs:
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                        help='Fix learning rate for the predictor')

    # dataset configs:
    parser.add_argument('--target-dataset', default='imagenet', type=str,
                        help='target dataset for training')

    # checkpoint saving configs:
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Number of training epochs before saving')
    parser.add_argument('--save-postfix', type=str, default='',
                        help='Postfix string for checkpoint dirname')
    parser.add_argument('--ngpus-per-node', type=int, default=-1,
                        help='Number of GPUs per node')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    assert(args.multiprocessing_distributed)
    args.distributed = True
    ngpus_per_node = (torch.cuda.device_count() if args.ngpus_per_node == -1
                      else args.ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print('Spawn {} processes in node {}'.format(
            ngpus_per_node, int(os.environ['SLURM_PROCID'])))
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        # main_worker(args.gpu, ngpus_per_node, args)
        raise NotImplementedError()
    return


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = int(os.environ['SLURM_PROCID']) * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print('Use GPU {} for training, Rank: {}, World Size: {}'.format(
            args.gpu, args.rank, args.world_size))
        torch.distributed.barrier()  # Sync so that all processes run to this func

    # suppress printing if not master
    if args.multiprocessing_distributed and args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'mobilenetv2':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2',
                               pretrained=False, num_classes=args.dim)
        model = simsiam.builder.SimSiam(
                model, args.dim, args.pred_dim, use_mobilenetv2=True)
    elif args.arch == 'swintiny':
        cfg_file = '/home/tu.343/Research/FractalPretrain/Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml'
        sys.path.insert(0, '/home/tu.343/Research/FractalPretrain/Swin-Transformer')
        from config import _C, _update_config_from_file
        from models import build_model
        sys.path.remove('/home/tu.343/Research/FractalPretrain/Swin-Transformer')
        config = _C.clone()
        config.MODEL.NUM_CLASSES = args.dim
        _update_config_from_file(config, cfg_file)
        model = build_model(config)
        model = simsiam.builder.SimSiam(
                model, args.dim, args.pred_dim, use_swintiny=True)
    else:
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch], args.dim, args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(),
                         'fix_lr': False},
                        {'params': model.module.predictor.parameters(),
                         'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])

            if not args.resume_only_weights:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> loaded checkpoint '{}' (only on weights)"
                      .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.target_dataset.startswith('imagenet'):
        traindir = os.path.join(args.data, 'train')
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)]

        if args.target_dataset == 'imagenet':
            train_dataset = MyImageFolder(
                traindir,
                simsiam.loader.TwoCropsTransform(
                    transforms.Compose(augmentation)))

        elif args.target_dataset == 'imagenet_downsampled':
            raise NotImplementedError()

    elif args.target_dataset.startswith('stylegan-oriented'):
        traindir = os.path.join(args.data, 'train')
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)]

        train_dataset = MyImageFolder(
                traindir,
                simsiam.loader.TwoCropsTransform(
                    transforms.Compose(augmentation)))

    elif args.target_dataset.startswith('multifractal_224'):
        fractal_args = parse_multifractal_setting(
                args.target_dataset)
        print(fractal_args)
        num_class = fractal_args.get('cls', 1000)
        num_systems = fractal_args.get('sys', 1000)
        assert(num_class == num_systems)
        total_images = 1000000
        per_class = int(total_images // num_class)
        max_num_objs = fractal_args.get('mnobjs', 2)
        color_mode = fractal_args.get('colormode', 'random')
        gen_num_augs = fractal_args.get('genaugs', 2)
        imgaug = fractal_args.get('imgaug', True)
        background = fractal_args.get('bg', True)
        image_size = fractal_args.get('imgsize', 224)
        print('num_class: {}, num_systems: {}, max_num_objs: {}, color_mode: {}, gen_num_augs: {}, imgaug: {}, background: {}'.format(
            num_class, num_systems, max_num_objs, color_mode, gen_num_augs, imgaug, background))
        assert(imgaug)

        if imgaug:
            image_level_augs = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])])
        else:
            image_level_augs = None

        generator = SelfSupGenerator(
                size=image_size,
                cache_size_per_class=4,    # for a 224x22 image size
                max_num_cached_class=1024,
                niter=100000,
                num_class=num_class,
                num_objects=(2, max_num_objs),
                color_mode=color_mode,
                num_augs=gen_num_augs,
                background=background)

        datamodule = SelfSupMultiFractalDataModule(
                data_dir='fractal_code/',
                data_file='ifs-1mil.pkl',
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                size=image_size,
                num_systems=num_systems,
                num_class=num_class,
                per_class=per_class,
                generator=generator,
                period=2, num_augs=2,
                transform=image_level_augs)
    else:
        raise NotImplementedError()

    if args.distributed:
        if args.target_dataset.startswith('imagenet'):
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset)
        elif args.target_dataset.startswith('stylegan-oriented'):
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset)
        elif args.target_dataset.startswith('multifractal_224'):
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    datamodule.data_train)
        else:
            raise NotImplementedError()
    else:
        train_sampler = None

    if args.target_dataset.startswith('imagenet'):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)

    elif args.target_dataset.startswith('stylegan-oriented'):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)

    elif args.target_dataset.startswith('multifractal_224'):
        train_loader = datamodule.train_dataloader(
                train_sampler=train_sampler)

    else:
        raise NotImplementedError()

    if args.target_dataset.startswith('multifractal_224'):
        print('SelfSupMultiFractals: num_class: {}, per_class: {}'.format(
            num_class, per_class))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        print('Learning Rate: {:.4f}'.format(optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            if (epoch+1) % args.save_freq == 0 or epoch == args.epochs-1:
                checkpoint_dir = 'MODELS/Distributed_SimSiam-{}_pretrain{}/{}'.format(
                        args.target_dataset, args.save_postfix, args.arch)
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(
                    checkpoint_dir, epoch + 1))
    return


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x1, x2, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            x1 = x1.cuda(args.gpu, non_blocking=True)
            x2 = x2.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=x1, x2=x2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), x1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
