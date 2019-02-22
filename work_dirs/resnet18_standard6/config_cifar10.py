# model settings
model = dict(type='models.resnet_cifar.resnet18', num_classes=10)
# dataset settings
data_root = '/home/gzhou/cifar10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
dataset_type = 'torchvision.datasets.CIFAR10'
data = dict(
    train=dict(
        type='custom_datasets.CustomDataset',
        base_dataset=dict(
            type=dataset_type,
            root=data_root,
            train=True,
            download=True,
            transform=dict(
                type='torchvision.transforms.Compose',
                transforms=[
                    dict(type='torchvision.transforms.RandomCrop',
                         size=32, padding=4),
                    dict(type='torchvision.transforms.RandomHorizontalFlip'),
                    dict(type='torchvision.transforms.ToTensor'),
                    dict(type='torchvision.transforms.Normalize',
                         mean=mean, std=std),
                ])),
        # subsample=dict(mode='uniform', sample_frac=0.5)),
        # logits=dict(path='./work_dirs/distill_train/resnet_ensemble_train_logits.pkl',
        #             temperature=1.0)),
        ),
    val=dict(
        type='custom_datasets.CustomDataset',
        base_dataset=dict(
            type=dataset_type,
            root=data_root,
            train=False,
            download=True,
            transform=dict(
                type='torchvision.transforms.Compose',
                transforms=[
                    dict(type='torchvision.transforms.ToTensor'),
                    dict(type='torchvision.transforms.Normalize',
                         mean=mean, std=std),
                ]))),
)
batch_size = 128

# optimizer and learning rate
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', gamma=0.2, step=[60, 120, 160])

# runtime settings
work_dir = './work_dirs/resnet18_standard6'
gpus = range(2)
dist_params = dict(backend='nccl')
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=10)
workflow = [('train', 1), ('val', 1)]
total_epochs = 200
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=10,  # log at every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
