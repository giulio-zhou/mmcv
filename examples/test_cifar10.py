import argparse
from argparse import ArgumentParser

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mobilenet
import mobilenetv2
import resnet_cifar

import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

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
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(x=data[0])

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

    if args.gpus == 1:
        model = DataParallel(model_cls(), device_ids=cfg.gpus).cuda()
        load_checkpoint(model, checkpoint)
        outputs = single_test(model, val_loader, args.show)
    else:
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
    print(outputs)

if __name__ == '__main__':
    main()
