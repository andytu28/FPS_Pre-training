import argparse
import sys
import random
import math
import os
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms, datasets
import resnet_cifar
from utils import Metric
from fractal_datamodule.selfsup_fractal_datamodule import SelfSupMultiFractalDataModule
from fractal_datamodule.datasets.selfsup_fractaldata import SelfSupGenerator

import moco.loader
import moco.builder
from resnet_cifar import BasicBlock as BasicBlockCifar
from resnet_cifar import ResNet as ResNetCifar
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_name', type=str, required=True,
                        help='Architecture name.')
    parser.add_argument('--expand', type=int, required=True,
                        help='Use expanded network.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Dataset name.')

    # Options for fractals
    parser.add_argument('--data_file', type=str, default='ifs-1mil.pklt',
                        help='IFS code data file.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Dataloader number of workers.')
    parser.add_argument('--num_class', type=int, default=100000,
                        help='Number of classes.')
    parser.add_argument('--num_systems', type=int, default=100000,
                        help='Numer of IFS systems.')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='Max number of epochs.')
    parser.add_argument('--max_num_objs', type=int, default=2,
                        help='Max number of objects in an image.')
    parser.add_argument('--color_mode', type=str, default='random',
                        help='Color mode for rendering fractals.')
    parser.add_argument('--gen_num_augs', type=int, default=2,
                        help='Number of augmentations for FPS.')
    parser.add_argument('--image_level_augs', default=True, action='store_true',
                        help='To use image level augmentations.')

    # moco specific configs:
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug_plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    args = parser.parse_args()

    # Make sure we are using MoCoV2
    assert(args.mlp and args.aug_plus and args.cos and args.moco_t == 0.2)

    if args.data_name == 'fractal':
        args.tar_num_augs = 2
        args.per_class = int(1000000 // args.num_class)
        assert(args.image_level_augs)
        assert(args.num_class == args.num_systems)
        args.data_dir = 'fractal_code/'
    elif args.data_name == 'stylegan-oriented':
        args.data_dir = 'data/stylegan-oriented'
    elif args.data_name == 'imagenet':
        args.data_dir = 'data/imagenet'
    else:
        raise NotImplementedError()

    args.image_size = 32
    args.lr = 0.03
    args.weight_decay = 1e-4
    args.momentum = 0.9
    args.batch_size = 256
    print('Arguments: ', args)
    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.max_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def run_train(args, model, train_loader, train_sampler):

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch_idx in range(1, args.max_epoch+1):
        train_sampler.set_epoch(epoch_idx-1)
        adjust_learning_rate(
                optimizer, epoch_idx-1, args)
        model.train()
        train_loss = Metric('train_loss')
        top1 = Metric('acc1')
        top5 = Metric('acc5')

        with tqdm(total=len(train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx),
                  disable=False,
                  dynamic_ncols=True,
                  ascii=True) as t:

            for batch_idx, (x1, x2, _) in enumerate(train_loader):
                x1 = x1.cuda(args.gpu, non_blocking=True)
                x2 = x2.cuda(args.gpu, non_blocking=True)

                # compute output
                output, target = model(im_q=x1, im_k=x2)
                loss = criterion(output, target)

                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                train_loss.update(loss.item(), x1.size(0))
                top1.update(acc1[0], x1.size(0))
                top5.update(acc5[0], x1.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix({
                        'loss': '{:.4f}'.format(train_loss.avg),
                        'lr':   '{:.4e}'.format(optimizer.param_groups[0]['lr']),
                        'top1': '{:.4f}'.format(top1.avg),
                        'top5': '{:.4f}'.format(top5.avg),
                })
                t.update(1)

        if ((epoch_idx % 10 == 0) or (epoch_idx == args.max_epoch)) and args.rank == 0:
            save_model(model.module, args, epoch_idx,
                       train_loss=train_loss.avg, top1=top1.avg, top5=top5.avg)
    return


def save_model(model, args, epoch_idx, train_loss, top1, top5):
    if args.data_name == 'fractal':
        output_path = (
            'MODELS/fractal_mocov2_cls-{}_sys-{}_mnobjs-{}_colormode-{}_genaugs-{}_{}_pretrain/{}'.format(
                args.num_class, args.num_systems,
                args.max_num_objs, args.color_mode, args.gen_num_augs,
                'w-imgaug' if args.image_level_augs else 'wo-imgaug',
                args.arch_name + ('' if args.expand == 1 else '_' + str(args.expand) + 'x')) +
            '/chkpt-ep{}.pth'.format(epoch_idx))

        # Add num_workers into the output path
        tokens = output_path.split('/')
        assert(len(tokens) == 4)
        output_path = os.path.join(
                tokens[0], tokens[1] + '_workers-{}'.format(args.num_workers),
                tokens[2], tokens[3])

    elif args.data_name == 'stylegan-oriented':
        output_path = 'MODELS/stylegan-oriented_mocov2_pretrain/{}/chkpt-ep{}.pth'.format(
                args.arch_name + ('' if args.expand == 1 else '_' + str(args.expand) + 'x'),
                epoch_idx)

    elif args.data_name == 'imagenet':
        output_path = 'MODELS/imagenet_mocov2_pretrain/{}/chkpt-ep{}.pth'.format(
                args.arch_name + ('' if args.expand == 1 else '_' + str(args.expand) + 'x'),
                epoch_idx)

    else:
        raise NotImplementedError()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state_dict = {}
    model_state_dict = model.state_dict()
    for k in list(model_state_dict.keys()):
        if (k.startswith('encoder_q') and not k.startswith('encoder_q.fc')):
            state_dict[k[len('encoder_q.'):]] = model_state_dict[k]
    torch.save({'state_dict': state_dict,
                'top1': '{:.4f}'.format(top1),
                'top5': '{:.4f}'.format(top5),
                'train_loss': '{:.4f}'.format(train_loss)}, output_path)
    return


def main():

    # Get the input arguments
    args = parse_arguments()

    # Spawn 1 process
    ngpus_per_node = 1
    args.world_size = 1
    mp.spawn(main_worker, nprocs=ngpus_per_node,
            args=(ngpus_per_node, args))
    return


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = 0
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:11260',
                            world_size=args.world_size,
                            rank=args.rank)
    print('Use GPU {} for training, Rank: {}, World Size: {}'.format(
        args.gpu, args.rank, args.world_size))
    torch.distributed.barrier()  # Sync so that all processes run to this func

    # suppress printing if not master
    if args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    # Get the dataloader
    if args.image_size == 32:
        cs_per_cls = 32
    else:
        raise NotImplementedError()

    if args.data_name == 'fractal':
        if args.image_level_augs:
            image_level_augs = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
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
                size=args.image_size,
                cache_size_per_class=cs_per_cls,
                max_num_cached_class=1024,
                niter=1000,
                num_class=args.num_class,
                num_objects=(2, args.max_num_objs),
                color_mode=args.color_mode,
                num_augs=args.gen_num_augs,
                background=False)

        datamodule = SelfSupMultiFractalDataModule(
                data_dir=args.data_dir,
                data_file=args.data_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                size=args.image_size,
                num_systems=args.num_systems,
                num_class=args.num_class,
                per_class=args.per_class,
                generator=generator,
                period=2, num_augs=args.tar_num_augs,
                transform=image_level_augs)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                datamodule.data_train)

        train_loader = datamodule.train_dataloader(
                train_sampler=train_sampler)

    elif args.data_name == 'stylegan-oriented':
        traindir = os.path.join(args.data_dir, 'train')
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.Resize(args.image_size), # Reisze to 32x32 to ensure we use the entire image
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)]

        train_dataset = MyImageFolder(
                traindir,
                moco.loader.TwoCropsTransform(
                    transforms.Compose(augmentation)))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)

    elif args.data_name == 'imagenet':
        traindir = os.path.join(args.data_dir, 'train')
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.Resize(args.image_size), # Reisze to 32x32 to ensure we use the entire image
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)]

        train_dataset = MyImageFolder(
                traindir,
                moco.loader.TwoCropsTransform(
                    transforms.Compose(augmentation)))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)

    else:
        raise NotImplementedError()

    # Get the model
    resnet_config = {'resnet20': [3, 3, 3],
                     'resnet32': [5, 5, 5],
                     'resnet44': [7, 7, 7],
                     'resnet56': [9, 9, 9]}
    def build_resnet(num_classes):
        return ResNetCifar(
                BasicBlockCifar, resnet_config[args.arch_name],
                num_classes=num_classes, expand=args.expand)

    print('==> creating model {}'.format(args.arch_name))
    model = moco.builder.MoCo(
            build_resnet,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
    print(model)

    if args.data_name == 'fractal':
        print('SelfSupMultiFractals: num_class: {}, per_class: {}'.format(
            args.num_class, args.per_class))

    # Run training loops
    run_train(args, model, train_loader, train_sampler)
    return


if __name__ == '__main__':
    main()
