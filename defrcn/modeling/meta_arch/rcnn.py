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
from defrcn.modeling.modules import Discriminator

from defrcn.utils.class_embedding import get_class_embedding
from defrcn.utils.class_name import get_class_name

__all__ = ["GeneralizedRCNN"]


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
        self.affine_rpn = AffineLayer(
            num_channels=self._SHAPE_["res4"].channels, bias=True
        )
        self.affine_rcnn = AffineLayer(
            num_channels=self._SHAPE_["res4"].channels, bias=True
        )
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
        proposal_losses, detector_losses, _, _ = self._forward_once_(
            batched_inputs, gt_instances
        )
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
            features_de_rpn = {
                k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features
            }
        proposals, proposal_losses = self.proposal_generator(
            images, features_de_rpn, gt_instances
        )

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {
                k: self.affine_rcnn(decouple_layer(features[k], scale))
                for k in features
            }
        results, detector_losses = self.roi_heads(
            images, features_de_rcnn, proposals, gt_instances
        )

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(self.cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        return lambda x: (x - pixel_mean) / pixel_std


@META_ARCH_REGISTRY.register()
class KDRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.out_visual_dim = self._SHAPE_["res4"].channels
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        if cfg.MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR:
            for p in self.roi_heads.box_predictor.parameters():
                p.requires_grad = False
            print("froze roi_box_predictor parameters")

<<<<<<< HEAD
        self.addition_model = cfg.MODEL.AUX.NAME
        self.inference_with_gt = cfg.MODEL.AUX.INFERENCE_WITH_GT
=======
        self.addition_model = cfg.MODEL.ADDITION.NAME
        self.inference_with_gt = cfg.MODEL.ADDITION.INFERENCE_WITH_GT
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.visual_dim = self._SHAPE_["res4"].channels

        self.teacher_training = cfg.MODEL.ADDITION.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ADDITION.STUDENT_TRAINING
        self.distill_on = cfg.MODEL.ADDITION.DISTILL_ON
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31

        if self.addition_model == "glove":
            self.semantic_dim = 300
        elif self.addition_model == "clip":
            self.semantic_dim = 512

        self.fixed_bg = True
        self.class_names = get_class_name(cfg)
        self.class_embedding = get_class_embedding(
            self.class_names, self.addition_model, include_bg=self.fixed_bg
        )
        if not self.fixed_bg:
            self.bg_feature_init = torch.randn(1, self.semantic_dim)
            self.bg_feature = nn.parameter.Parameter(
                self.bg_feature_init.clone(), requires_grad=True
            )
            
        self.sem2vis_proj = nn.ModuleDict()
        for scale in self._SHAPE_:
            self.sem2vis_proj[scale] = nn.Sequential(
                nn.Conv2d(
                    self.semantic_dim * self.num_classes,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
            )
        self.vis_adapter = nn.ModuleDict()
        for scale in self._SHAPE_:
            self.vis_adapter[scale] = nn.Sequential(
                nn.Conv2d(
                    self._SHAPE_[scale].channels,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self._SHAPE_[scale].channels,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
            )

<<<<<<< HEAD
        self.to(self.device)

    def forward(self, batched_inputs):
        if not self.training: 
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        kd_loss, proposal_losses, detector_losses, _, _ = self._forward_once_(
            batched_inputs, gt_instances
=======
        self.combined2vis_proj = nn.Conv2d(
            self.semantic_dim + self.visual_dim,
            self.visual_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        )
        losses = {}
        losses.update(kd_loss)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        print(losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        gt_instances = None
        if self.inference_with_gt:
            assert "instances" in batched_inputs[0]
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

<<<<<<< HEAD
        _, _, _, results, image_sizes = self._forward_once_(
            batched_inputs, gt_instances
=======
        self.student_adapter = nn.Sequential(
            nn.ConvTranspose2d(
                self.visual_dim,
                self.visual_dim + self.semantic_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.visual_dim + self.semantic_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.visual_dim + self.semantic_dim,
                self.visual_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        )
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

<<<<<<< HEAD
        kd_loss, features, teacher_features = self._distillate(features, gt_instances)
        features_de_rpn = features
        teacher_features_de_rpn = teacher_features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {
                k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features
            }
            teacher_features_de_rpn = decouple_layer(teacher_features, scale)
            
        proposals, proposal_losses = self.proposal_generator(
            images,
            features_de_rpn,
            gt_instances,
            teacher_features_de_rpn
        )

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {
                k: self.affine_rcnn(decouple_layer(features[k], scale))
                for k in features
            }
        results, detector_losses = self.roi_heads(
            images,
            features_de_rcnn,
            proposals,
            gt_instances,
=======
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.visual_dim, self.visual_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.visual_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.visual_dim // 2, self.visual_dim // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.visual_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.visual_dim // 4, 1, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        )

        return kd_loss, proposal_losses, detector_losses, results, images.image_sizes

    def _expand_bbox(self, gt_box, H, W, S, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / S
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, x_c - w / 2))
        y1 = int(max(0, y_c - h / 2))
        x2 = int(min(W, x_c + w / 2))
        y2 = int(min(H, y_c + h / 2))
        return x1, y1, x2, y2

    def _generate_semantic_features_overlap(self, B, H, W, gt_instances, stride, expand_rate):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]

        features = torch.zeros((B, H, W, self.semantic_dim), device=self.device)
        features[:, :, :] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(
            zip(gt_boxes, gt_classes)
        ):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, H, W, stride, expand_rate)
                features[idx, y1:y2, x1:x2] = self.class_embedding[gt_class]

        # (B, H, W, channels) -> (B, channels, H, W)
        features = features.permute(0, 3, 1, 2)

        return features

    def _generate_semantic_features(self, B, H, W, gt_instances, scale, expand_rate):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        C = self.num_classes    
        D = self.semantic_dim
        S = self._SHAPE_[scale].stride

        features = torch.zeros((B, H, W, C, D), device=self.device)
        for idx in range(C):
            features[:, :, :, idx] = -self.class_embedding[idx]
        features[:, :, :, C - 1] = -torch.mean(self.class_embedding, dim=0)
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(
            zip(gt_boxes, gt_classes)
        ):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, H, W, S, expand_rate)
                features[idx, y1:y2, x1:x2, gt_class] = self.class_embedding[gt_class]

        # (B, H, W, C, D) -> (B, H, W, C * D)
        features = features.view(B, H, W, C * D).contiguous()
        # (B, H, W, C * D) -> (B, C * D, H, W)
        features = features.permute(0, 3, 1, 2)
        # B, C * D, H, W -> B, 1024, H, W
        features = self.sem2vis_proj[scale](features)

        return features

    def _add_semantic_features(self, visual_features, gt_instances):
        B, _, H, W = visual_features.shape
        stride = self._SHAPE_["res4"].stride
        expand_rate = 1.2

        semantic_features = self._generate_semantic_features(
            B, H, W, gt_instances, stride, expand_rate
        )
        # (B, H, W, 1024) + (B, H, W, 300) -> (B, H, W, 1324)
        combined_features = torch.cat([visual_features, semantic_features], dim=1)
        # (B, 1324, H, W) -> (B, 1024, H, W)
        combined_features = self.combined2vis_proj(combined_features)
        return combined_features

    def _distillate(self, visual_features, gt_instances):
        losses = {}
        for scale in visual_features:
            if scale == "res2":
                expand_rate = 1.0
            elif scale == "res3":
                expand_rate = 1.0
            elif scale == "res4":
                expand_rate = 1.0
            B, _, H, W = visual_features[scale].shape
            adapted_visual_features = self.vis_adapter[scale](visual_features[scale])
            # B, 300, H, W -> B, 300, 1
            # projected_visual_features = self.vis2sem_proj[scale](visual_features[scale])

            if self.training:
                semantic_features = self._generate_semantic_features(
                    B, H, W, gt_instances, scale, expand_rate
                )
                # loss = torch.mean(1 - F.cosine_similarity(
                #     adapted_visual_features, semantic_features, dim=1))
                loss = F.mse_loss(
                            F.normalize(adapted_visual_features, p=2),
                            F.normalize(semantic_features, p=2),
                        )
                alpha = 1
                losses.update(
                    {
                        f"loss_rpn_{scale}": alpha * loss
                    }
                )
            #     visual_features[scale] = torch.add(
            #         visual_features[scale], semantic_features
            #     )
            # else:
            #     visual_features[scale] = torch.add(
            #         visual_features[scale], adapted_visual_features
            #     )
                semantic_features = torch.add(visual_features[scale], semantic_features)
                visual_features[scale] = torch.add(visual_features[scale], adapted_visual_features)
                
                return losses, visual_features, semantic_features
            
            visual_features[scale] = torch.add(visual_features[scale], adapted_visual_features)
        
        return losses, visual_features, None

@META_ARCH_REGISTRY.register()
class KDGANRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.out_visual_dim = self._SHAPE_["res4"].channels
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        if cfg.MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR:
            for p in self.roi_heads.box_predictor.parameters():
                p.requires_grad = False
            print("froze roi_box_predictor parameters")

        self.addition_model = cfg.MODEL.AUX.NAME
        self.inference_with_gt = cfg.MODEL.AUX.INFERENCE_WITH_GT
        # self.teacher_training = cfg.MODEL.AUX.TEACHER_TRAINING
        # self.student_training = cfg.MODEL.AUX.STUDENT_TRAINING
        # self.distill_on = cfg.MODEL.AUX.DISTILL_ON
        # self.distill_features = cfg.MODEL.AUX.DISTILL_FEATURES

        if self.addition_model == "glove":
            self.semantic_dim = 300
        elif self.addition_model == "clip":
            self.semantic_dim = 512

        self.fixed_bg = True
        self.class_names = get_class_name(cfg)
        self.class_embedding = get_class_embedding(
            self.class_names, self.addition_model, include_bg=self.fixed_bg
        )
        if not self.fixed_bg:
            self.bg_feature_init = torch.randn(1, self.semantic_dim)
            self.bg_feature = nn.parameter.Parameter(
                self.bg_feature_init.clone(), requires_grad=True
            )

        ### GAN
        # self.combined2vis_proj = nn.Conv2d(
        #     self.semantic_dim + self.out_visual_dim,
        #     self.out_visual_dim,
        #     kernel_size=1,
        # )

        # self.generator = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         self.out_visual_dim,
        #         self.out_visual_dim + self.semantic_dim,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.BatchNorm2d(self.out_visual_dim + self.semantic_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         self.out_visual_dim + self.semantic_dim,
        #         self.out_visual_dim,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.Tanh(),
        # )

        # self.discriminator = nn.Sequential(
        #     nn.Conv2d(self.out_visual_dim, self.out_visual_dim // 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.out_visual_dim // 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.out_visual_dim // 2, self.out_visual_dim // 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.out_visual_dim // 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.out_visual_dim // 4, 1, 4, 1, 0, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Sigmoid()
        # )
        ###

        # self.vis2sem_proj = nn.ModuleDict()
        # for scale in self._SHAPE_:
        #     self.vis2sem_proj[scale] = nn.Sequential(
        #         nn.Conv2d(
        #             self._SHAPE_[scale].channels,
        #             self.semantic_dim,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #         ),
        #         nn.ReLU(inplace=True),
        #     )

        self.sem2vis_proj = nn.ModuleDict()
        for scale in self._SHAPE_:
            self.sem2vis_proj[scale] = nn.Sequential(
                nn.Conv2d(
                    self.semantic_dim * self.num_classes,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
            )
        self.vis_adapter = nn.ModuleDict()
        for scale in self._SHAPE_:
            self.vis_adapter[scale] = nn.Sequential(
                nn.Conv2d(
                    self._SHAPE_[scale].channels,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self._SHAPE_[scale].channels,
                    self._SHAPE_[scale].channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self._SHAPE_[scale].channels),
                nn.ReLU(inplace=True),
            )

        self.to(self.device)

    def _init_weight(self, module):
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.Linear)
        ):
            # nn.init.normal_(module.weight, mean=0, std=0.002)
            print("init", module.__class__.__name__)
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, batched_inputs):
        if not self.training: 
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        kd_loss, proposal_losses, detector_losses, _, _ = self._forward_once_(
            batched_inputs, gt_instances
        )
        losses = {}
        losses.update(kd_loss)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # print(losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        gt_instances = None
        if self.inference_with_gt:
            assert "instances" in batched_inputs[0]
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        _, _, _, results, image_sizes = self._forward_once_(
            batched_inputs, gt_instances
        )
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

        kd_loss, features = self._distillate(features, gt_instances)
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {
                k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features
            }
        proposals, proposal_losses = self.proposal_generator(
            images,
            features_de_rpn,
            gt_instances,
        )

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {
                k: self.affine_rcnn(decouple_layer(features[k], scale))
                for k in features
            }
        results, detector_losses = self.roi_heads(
            images,
            features_de_rcnn,
            proposals,
            gt_instances,
        )

        return kd_loss, proposal_losses, detector_losses, results, images.image_sizes

    def _forward_once_gan(self, batched_inputs, gt_instances=None):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        adv_losses = {}
        real_features = None
        if self.teacher_training:
            features = {
                f: self._add_semantic_features(features[f], gt_instances)
                for f in features
            }

        if self.student_training:
<<<<<<< HEAD
            features = {f: self.generator(features[f]) for f in features}
=======
            features = {f: self.student_adapter(features[f]) for f in features}
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
            if self.distill_on and self.training:
                teacher_features = {
                    f: self._generate_semantic_features(features[f], gt_instances)
                    for f in features
                }
                for f in features:
                    gen_loss, disc_loss = self._forward_gan(
                        teacher_features[f], features[f]
                    )

                adv_losses.update({"loss_adv_d": disc_loss, "loss_adv_g": gen_loss})

                real_features = teacher_features

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {
                k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features
            }
        proposals, proposal_losses = self.proposal_generator(
<<<<<<< HEAD
            images, features_de_rpn, gt_instances, real_features
=======
            images, features_de_rpn, gt_instances, 
            # real_features
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        )

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {
                k: self.affine_rcnn(decouple_layer(features[k], scale))
                for k in features
            }
        results, detector_losses = self.roi_heads(
            images, features_de_rcnn, proposals, gt_instances, real_features
        )

        return adv_losses, proposal_losses, detector_losses, results, images.image_sizes

<<<<<<< HEAD
    def _merge_losses(self, adv_losses, detector_losses, alpha=10):
        loss_kd = detector_losses.pop("loss_kd")
        adv_losses.update({"loss_adv_g": adv_losses["loss_adv_g"] + alpha * loss_kd})

        return adv_losses, detector_losses

=======
    def _forward_discriminator(self, features):
        # print("origin", features.shape)
        logits = self.discriminator(features)
        # print("logits", logits.shape)
        return logits
    
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
    def _forward_gan(self, real_features, fake_features):
        logit_real = self.discriminator(real_features)
        logit_fake = self.discriminator(fake_features)

        gen_loss = F.mse_loss(fake_features, torch.ones_like(fake_features))
        disc_loss_real = F.mse_loss(logit_real, torch.ones_like(logit_real))
        disc_loss_fake = F.mse_loss(logit_fake, torch.zeros_like(logit_fake))
        disc_loss = disc_loss_real + disc_loss_fake

        return gen_loss, disc_loss

    def _expand_bbox(self, gt_box, H, W, S, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / S
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, x_c - w / 2))
        y1 = int(max(0, y_c - h / 2))
        x2 = int(min(W, x_c + w / 2))
        y2 = int(min(H, y_c + h / 2))
        return x1, y1, x2, y2

    def _generate_semantic_features_overlap(
        self, B, H, W, gt_instances, stride, expand_rate
    ):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]

        features = torch.zeros((B, H, W, self.semantic_dim), device=self.device)
        features[:, :, :] = self.bg_feature
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(
            zip(gt_boxes, gt_classes)
        ):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, H, W, stride, expand_rate)
                features[idx, y1:y2, x1:x2] = self.class_embedding[gt_class]

        # (B, H, W, channels) -> (B, channels, H, W)
        features = features.permute(0, 3, 1, 2)

        return features

    def _generate_semantic_features(self, B, H, W, gt_instances, scale, expand_rate):
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        C = self.num_classes    
        D = self.semantic_dim
        S = self._SHAPE_[scale].stride

        features = torch.zeros((B, H, W, C, D), device=self.device)
        for idx in range(C):
            features[:, :, :, idx] = -self.class_embedding[idx]
        features[:, :, :, C - 1] = -torch.mean(self.class_embedding, dim=0)
        for idx, (gt_boxes_per_img, gt_classes_per_img) in enumerate(
            zip(gt_boxes, gt_classes)
        ):
            for gt_box, gt_class in zip(gt_boxes_per_img, gt_classes_per_img):
                x1, y1, x2, y2 = self._expand_bbox(gt_box, H, W, S, expand_rate)
                features[idx, y1:y2, x1:x2, gt_class] = self.class_embedding[gt_class]

        # (B, H, W, C, D) -> (B, H, W, C * D)
        features = features.view(B, H, W, C * D).contiguous()
        # (B, H, W, C * D) -> (B, C * D, H, W)
        features = features.permute(0, 3, 1, 2)
        # B, C * D, H, W -> B, 1024, H, W
        features = self.sem2vis_proj[scale](features)

        return features

    def _add_semantic_features(self, visual_features, gt_instances):
        B, _, H, W = visual_features.shape
        stride = self._SHAPE_["res4"].stride
        expand_rate = 1.2

        semantic_features = self._generate_semantic_features(
            B, H, W, gt_instances, stride, expand_rate
        )
        # (B, H, W, 1024) + (B, H, W, 300) -> (B, H, W, 1324)
        combined_features = torch.cat([visual_features, semantic_features], dim=1)
        # (B, 1324, H, W) -> (B, 1024, H, W)
        combined_features = self.combined2vis_proj(combined_features)
        return combined_features

    def _distillate(self, visual_features, gt_instances):
        losses = {}
        for scale in visual_features:
            if scale == "res2":
                expand_rate = 1.0
            elif scale == "res3":
                expand_rate = 1.0
            elif scale == "res4":
                expand_rate = 1.0
            B, _, H, W = visual_features[scale].shape
            adapted_visual_features = self.vis_adapter[scale](visual_features[scale])
            # visual_features[scale] = torch.add(
            #     visual_features[scale], adapted_visual_features
            # )

            # B, 300, H, W -> B, 300, 1
            # projected_visual_features = self.vis2sem_proj[scale](visual_features[scale])

            if self.training:
                semantic_features = self._generate_semantic_features(
                    B, H, W, gt_instances, scale, expand_rate
                )
                # loss = torch.mean(1 - F.cosine_similarity(
                #     adapted_visual_features, semantic_features, dim=1))
                loss = F.mse_loss(
                            F.normalize(adapted_visual_features, p=2),
                            F.normalize(semantic_features, p=2),
                        )
                alpha = 1
                losses.update(
                    {
                        f"loss_rpn_{scale}": alpha * loss
                    }
                )
                visual_features[scale] = torch.mul(
                    visual_features[scale], semantic_features
                )
            else:
                visual_features[scale] = torch.mul(
                    visual_features[scale], adapted_visual_features
                )
        return losses, visual_features

@META_ARCH_REGISTRY.register()
class DCGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.to(self.device)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        assert "fs_class" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        fs_class = [x["fs_class"] for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances, fs_class)
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

    def _forward_once_(self, batched_inputs, gt_instances=None, fs_class=None):

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
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances, fs_class)

        return proposal_losses, detector_losses, results, images.image_sizes

