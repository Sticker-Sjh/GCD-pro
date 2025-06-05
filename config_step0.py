_base_ = '../_base_/gdino_inc_base_coco.py'
pretrained = r'E:\sjh\model\swin_tiny_patch4_window7_224.pth'

dataset_type = 'CocoIncDataset'
data_root = 'E:/sjh2/GCD-main/data/coco/'
start = 0
end = 1

model = dict(
    type='GroundingDINO_inc',
    num_queries=900,
    backbone=dict(
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    bbox_head=dict(trunc_class=[start, end]),   # only used in CdnQueryGenerator for training
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/1+1+1+1/train_0.json',
        start=start,
        end=end))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/1+1+1+1/val_0.json',
        start=start,
        end=end))

val_evaluator = dict(ann_file=data_root + 'annotations/1+1+1+1/val_0.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

# learning policy
max_epochs = 12