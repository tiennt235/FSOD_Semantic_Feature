import torch
import logging
from torch import nn
import torch.nn.functional as F
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
# from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.modeling.proposal_generator import build_proposal_generator
from ..backbone import build_backbone
from ..proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer

from defrcn.modeling.roi_heads import build_roi_heads
from defrcn.data.builtin_meta import PASCAL_VOC_ALL_CATEGORIES, PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES
from defrcn.data.builtin_meta import _get_coco_fewshot_instances_meta
from defrcn.utils.get_semantic_features import get_semantic_features

__all__ = ["GeneralizedRCNN", "GeneralizedSemanticRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
    
    
@META_ARCH_REGISTRY.register()
class GeneralizedSemanticRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.rpn_addition = cfg.MODEL.RPN.ADDITION
        self.features_channels = self._SHAPE_["res4"].channels
        self.class_names = self._get_class_name(cfg)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.semantic_dim = cfg.ADDITION.SEMANTIC_DIM
        self.to_rpn_input_proj = nn.Linear(self.features_channels + self.semantic_dim, self.features_channels).to(self.device)
        self.semantic_features = get_semantic_features(self.class_names, model="glove", semantic_dim=self.semantic_dim).to(self.device)
        self.bg_feature_init = torch.randn(1, self.semantic_dim)
        self.bg_feature = nn.parameter.Parameter(self.bg_feature_init.clone(), requires_grad=True).to(self.device)
        self.teacher_training = cfg.MODEL.DISTILLATION.TEACHER_TRAINING
        if self.teacher_training == False:
            for m in [self.roi_heads.box_predictor, self.roi_heads.addition]:
                for p in m.parameters():
                    p.requires_grad = False
            
        
    def _get_class_name(self, cfg):
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

    def _forward_once_(self, batched_inputs, gt_instances=None):
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if self.rpn_addition:
            features = {k: self._add_semantic_features(features[k], gt_instances, self.semantic_features)
                        for k in features}
            
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)

        return proposal_losses, detector_losses, results, images.image_sizes
    
    def inference(self, batched_inputs):
        assert not self.training
        
        gt_instances = None
        if self.teacher_training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        _, _, results, image_sizes = self._forward_once_(batched_inputs, gt_instances)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _expand_bbox(self, gt_box, max_size, stride, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / stride
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, (x_c - w/2) // 1)) # x // 1 = floor(x)
        y1 = int(max(0, (int(y_c - h/2) // 1))) 
        x2 = int(min(max_size[1], (x_c + w/2) // 1 + 1)) # x // 1 = ceil(x)
        y2 = int(min(max_size[0], (y_c + h/2) // 1 + 1))
        return x1, y1, x2, y2
    
    def _add_semantic_features(self, vis_features, gt_instances, semantic_features, stride=16):
        # (B, channels, H, W) -> (B, H, W, channels)
        vis_features = vis_features.permute(0, 2, 3, 1)
        
        max_size = vis_features.shape[1:3]
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        
        features = torch.zeros((len(gt_instances), max_size[0], max_size[1], self.semantic_dim), 
                               device=self.device)
        features[:,:,:] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(zip(gt_boxes, gt_classes)):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, max_size, stride, 1.5)
                features[idx, x1:x2, y1:y2] = semantic_features[gt_class]
        
        features = torch.cat((vis_features, features), dim=-1)
        features = self.to_rpn_input_proj.to(self.device)(features)
        
        # (B, H, W, channels) -> (B, channels, H, W)
        features = features.permute(0, 3, 1, 2)
        
        return features

        
        
