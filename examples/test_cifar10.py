import argparse
import pickle
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mobilenet
import mobilenetv2
import resnet_cifar

import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Test CIFAR10 models')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file or checkpoint dir')
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

    cfg = mmcv.Config.fromfile(args.config)
    checkpoint = args.checkpoint

    num_workers = cfg.data_workers * len(cfg.gpus)
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    val_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # sampler=val_sampler,
        num_workers=num_workers)

    # build model
    if 'resnet' in cfg.model:
        model_cls = getattr(resnet_cifar, cfg.model)
    elif cfg.model == 'mobilenet':
        model_cls = mobilenet.MobileNet
    elif cfg.model == 'mobilenetv2':
        model_cls = mobilenetv2.MobileNetV2

    # Need higher ulimit for data loaders.
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

    if args.gpus == 1:
        model = DataParallel(model_cls(), device_ids=cfg.gpus).cuda()
        load_checkpoint(model, checkpoint)
        outputs = single_test(model, val_loader, args.show)
        targets = torch.LongTensor([x[1] for x in outputs]).cuda()
        outputs = torch.cat([x[0] for x in outputs])
        with open(args.out, 'wb') as f:
            pickle.dump(outputs, f)
    else:
        # NOTE: Parallel inference requires the data to be explicitly swapped to
        #       cpu (add a .cpu() call to the result in parallel_test.py).
        # model_args = cfg.model.copy()
        # model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        outputs = parallel_test(
            model_cls,
            {},
            checkpoint,
            val_dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)
        targets = torch.LongTensor([val_dataset[i][1]
                                    for i in range(len(val_dataset))])
        outputs = torch.cat(outputs).cpu()
        with open(args.out, 'wb') as f:
            pickle.dump(outputs, f)
    print(accuracy(outputs, targets, topk=(1,)))

if __name__ == '__main__':
    main()
