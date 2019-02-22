import argparse
import os
import pickle
import sys
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import custom_datasets
import models.mobilenet
import models.mobilenetv2
import models.resnet_cifar

import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict

def eval_mmcv_str(obj_type):
    if obj_type in sys.modules:
        obj_type = sys.modules[obj_type]
    else:
        # Assume the last part is a function/member name.
        elems = obj_type.split('.')
        module, attr = '.'.join(elems[:-1]), elems[-1]
        obj_type = getattr(sys.modules[module], attr)
    return obj_type

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
        obj_type = eval_mmcv_str(obj_type)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Test CIFAR10 models')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file or list of checkpoint dirs')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args

def _data_func(data, device_id):
    data = data[0].unsqueeze(0).cuda(device_id)
    return dict(x=data)

def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        data, label = data
        with torch.no_grad():
            result = model(**dict(x=data))
        results.append((result, label))

        batch_size = data.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

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

def main():
    args = parse_args()

    checkpoints = args.checkpoint.split(',') 
    if os.path.isdir(checkpoints[0]):
        configs = [mmcv.Config.fromfile(checkpoints[i] + '/' + args.config)
                   for i in range(len(checkpoints))]
        cfg = configs[0]
    else:
        cfg = mmcv.Config.fromfile(args.config)
        configs = [cfg]
    checkpoint = args.checkpoint

    val_dataset = deep_recursive_obj_from_dict(cfg.data.val)
    per_model_outputs = []
    for i, (checkpoint, curr_cfg) in enumerate(zip(checkpoints,
                                                   configs)):
        # build model
        model_cls = eval_mmcv_str(curr_cfg.model['type'])
        model_args = curr_cfg.model
        model_args.pop('type')

        # Need higher ulimit for data loaders.
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

        if os.path.isdir(checkpoints[0]):
            checkpoint_path = checkpoint + '/latest.pth'
            pkl_path = checkpoint + '/' + args.out
        else:
            checkpoint_path, pkl_path = checkpoint, args.out

        # Run model if results don't already exist.
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                outputs = pickle.load(f)
            targets = torch.LongTensor([val_dataset[i][1]
                                        for i in range(len(val_dataset))])
        elif args.gpus == 1:
            num_workers = curr_cfg.data_workers * len(cfg.gpus)
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                # sampler=val_sampler,
                num_workers=num_workers)
            # Build and run model.
            model = DataParallel(model, device_ids=range(args.gpus)).cuda()
            load_checkpoint(model_cls(**model_args), checkpoint_path)
            outputs = single_test(model, val_loader, args.show)
            targets = torch.LongTensor([x[1] for x in outputs]).cuda()
            outputs = torch.cat([x[0] for x in outputs])
            with open(pkl_path, 'wb') as f:
                pickle.dump(outputs, f)
        else:
            # NOTE: Parallel inference requires the data to be explicitly swapped to
            #       cpu (add a .cpu() call to the result in parallel_test.py).
            # model_args = cfg.model.copy()
            # model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
            def model_fn(**kwargs):
                return model
            outputs = parallel_test(
                model_cls,
                model_args,
                checkpoint_path,
                val_dataset,
                _data_func,
                range(args.gpus),
                workers_per_gpu=args.proc_per_gpu)
            targets = torch.LongTensor([val_dataset[i][1]
                                        for i in range(len(val_dataset))])
            outputs = torch.cat(outputs).cpu()
            with open(pkl_path, 'wb') as f:
                pickle.dump(outputs, f)
        print(checkpoint, accuracy(outputs, targets, topk=(1,)))
        per_model_outputs.append(outputs)

    # Naive averaging.
    avg = torch.mean(torch.stack(per_model_outputs), 0)
    print("Naive Averaging", accuracy(avg, targets, topk=(1,)))
    with open('avg.pkl', 'wb') as f:
        pickle.dump(avg, f)

if __name__ == '__main__':
    main()
