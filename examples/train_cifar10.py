import logging
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import Runner, DistSamplerSeedHook
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import custom_datasets
import models.mobilenet
import models.mobilenetv2
import models.resnet_cifar

import mmcv
import os.path as osp
import shutil


def deep_recursive_obj_from_dict(info):
    """Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    # TODO: This does not support object dicts nested in non-object dicts.
    args = info.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if obj_type in sys.modules:
            obj_type = sys.modules[obj_type]
        else:
            # Assume the last part is a function/member name.
            elems = obj_type.split('.')
            module, attr = '.'.join(elems[:-1]), elems[-1]
            obj_type = getattr(sys.modules[module], attr)
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    evaluated_args = {}
    for argname, argval in args.items():
        print(argname, type(argval))
        if isinstance(argval, dict) and 'type' in argval:
            evaluated_args[argname] = deep_recursive_obj_from_dict(argval)
        elif type(argval) == list or type(argval) == tuple:
            # Transform each dict in the list, else simply append.
            transformed_list = []
            for elem in argval:
                if isinstance(elem, dict):
                    transformed_list.append(deep_recursive_obj_from_dict(elem))
                else:
                    transformed_list.append(elem)
            evaluated_args[argname] = type(argval)(transformed_list)
        else:
            evaluated_args[argname] = argval
    print(obj_type)
    return obj_type(**evaluated_args)

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode):
    img, label, logits = data
    label = label.cuda(non_blocking=True)
    logits = logits.cuda(non_blocking=True)
    pred = model(img)
    if len(logits.size()) > 1:
        loss = F.kl_div(F.log_softmax(pred, 1), F.softmax(logits, 1))
    else:
        loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc_top1'] = acc_top1.item()
    log_vars['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger


def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    logger = get_logger(cfg.log_level)

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
        logger.info('Disabled distributed training.')
    else:
        dist = True
        init_dist(**cfg.dist_params)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank != 0:
            logger.setLevel('ERROR')
        logger.info('Enabled distributed training.')

    # build datasets and dataloaders
    train_dataset = deep_recursive_obj_from_dict(cfg.data.train)
    val_dataset = deep_recursive_obj_from_dict(cfg.data.val)
    if dist:
        num_workers = cfg.data_workers
        assert cfg.batch_size % world_size == 0
        batch_size = cfg.batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, world_size, rank)
        val_sampler = DistributedSampler(val_dataset, world_size, rank)
        shuffle = False
    else:
        num_workers = cfg.data_workers * len(cfg.gpus)
        batch_size = cfg.batch_size
        train_sampler = None
        val_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers)

    # build model
    model = deep_recursive_obj_from_dict(cfg.model)
    if dist:
        model = DistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model, device_ids=cfg.gpus).cuda()

    # build runner and register hooks
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level)
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)
    if dist:
        runner.register_hook(DistSamplerSeedHook())

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    # Create work_dir if necessary and copy config file.
    if mmcv.is_str(cfg.work_dir):
        work_dir = osp.abspath(cfg.work_dir)
        mmcv.mkdir_or_exist(work_dir)
        shutil.copy(args.config, work_dir)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
