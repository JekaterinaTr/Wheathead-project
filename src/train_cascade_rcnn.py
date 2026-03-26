# -*- coding: utf-8 -*-
# ---------------------------------------------------------
# Cascade R-CNN — Wheat Heads
# YOLO / RF-DETR style augmentations
# MMDetection 2.25.0
# ---------------------------------------------------------

_base_ = '../_base_/default_runtime.py'

# -------------------------
# Dataset
# -------------------------
dataset_type = 'CocoDataset'
data_root = 'D:/Projects/Project1_wheatheads/project1_dataset/'

classes = ('wheat_head',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(832, 832), (896, 896)],  multiscale_mode='range', keep_ratio=True),  # use MMDetection Resize
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(
        #type='RandomCrop',
        #crop_size=(512, 512),
        #allow_negative_crop=True
    #),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/annotations/instances_train.json',
        img_prefix=f'{data_root}/images/train/',
        pipeline=train_pipeline,
        filter_empty_gt=False 
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/annotations/instances_val.json',
        img_prefix=f'{data_root}/images/val/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/annotations/instances_val.json',
        img_prefix=f'{data_root}/images/val/',
        pipeline=test_pipeline
    ),
    persistent_workers=False
)

# -------------------------
# Model
# -------------------------
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages = 1
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.75, 1.0, 1.25],
            strides=[4, 8, 16, 32]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True),
        loss_bbox=dict(type='L1Loss')
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 0.75, 0.5],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
        ]
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    match_low_quality=True,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True
                ),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    match_low_quality=True,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True
                ),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.65,
                    neg_iou_thr=0.65,
                    min_pos_iou=0.65,
                    match_low_quality=True,
                    ignore_iof_thr=-1
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True
                ),
                pos_weight=-1,
                debug=False
            ),
        ]
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=500,
            nms=dict(type='soft_nms', iou_threshold=0.7, sigma=0.5),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.6, sigma=0.5),
            max_per_img=300
        )
    )
)


# -------------------------
# Optimizer
# -------------------------
optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=1e-4
)

optimizer_config = dict(grad_clip=None)

# Learning rate scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=True,
    min_lr=1e-5,
    warmup=None
)

# Runner
runner = dict(type='EpochBasedRunner', max_epochs=20)


# -------------------------
# Training settings
# -------------------------

val_interval = 1
randomness = dict(seed=0)
fp16 = None

# -------------------------
# Checkpoint & work directory
# -------------------------
work_dir = 'D:/mmdetection_checkpoints/cascade_wheat_yolo_style'

checkpoint_config = dict(
    interval=1,          # save every epoch
    max_keep_ckpts=20
          
)

evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP_75',
    rule='greater'
)
# -------------------------
# Pretrained weights
# -------------------------
load_from = r"C:/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
#resume_from = 'D:/mmdetection_checkpoints/cascade_wheat_yolo_style/epoch_8.pth'
#load_from = None