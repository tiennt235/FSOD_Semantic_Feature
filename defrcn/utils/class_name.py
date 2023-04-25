from defrcn.data.builtin_meta import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES
from defrcn.data.builtin_meta import _get_coco_fewshot_instances_meta

def get_class_name(cfg):
    dataset = cfg.DATASETS.TRAIN[0]
    if 'voc' in dataset:
        if 'base' in dataset:
            classes = PASCAL_VOC_BASE_CATEGORIES[int(dataset.split('_')[-1][-1])]
        if 'novel' in dataset:
            classes = PASCAL_VOC_NOVEL_CATEGORIES[int(dataset.split('_')[-3][-1])]
        if 'all' in dataset:
            classes = PASCAL_VOC_ALL_CATEGORIES[int(dataset.split('_')[-3][-1])]
    if 'coco' in dataset:
        ret = _get_coco_fewshot_instances_meta()
        if 'base' in dataset:
            classes = ret["base_classes"]
        if 'novel' in dataset:
            classes = ret["novel_classes"]
        if 'all' in dataset:
            classes = ret["thing_classes"]
    return classes