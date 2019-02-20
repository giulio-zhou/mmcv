# model settings
model = 'resnet18'
# dataset settings
data_root = '/home/giuliozhou/cifar10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 128

# optimizer and learning rate
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = dict(type='Adam')
optimizer_config = dict(grad_clip=None)
# lr_config = dict(policy='step', gamma=0.2, step=[60, 120, 160])
lr_config = dict(policy='fixed')

# runtime settings
work_dir = './work_dirs/resnet18_adam'
gpus = range(2)
dist_params = dict(backend='nccl')
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=5)  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
total_epochs = 200
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(
    # interval=50,  # log at every 50 iterations
    interval=10,  # log at every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
