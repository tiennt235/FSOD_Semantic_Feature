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

        if cfg.MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR:
            for p in self.roi_heads.box_predictor.parameters():
                p.requires_grad = False
            print("froze roi_box_predictor parameters")

        self.addition_model = cfg.MODEL.ADDITION.NAME
        self.inference_with_gt = cfg.MODEL.ADDITION.INFERENCE_WITH_GT
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.visual_dim = self._SHAPE_["res4"].channels

        self.teacher_training = cfg.MODEL.ADDITION.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ADDITION.STUDENT_TRAINING
        self.distill_on = cfg.MODEL.ADDITION.DISTILL_ON

        if self.addition_model == "glove":
            self.semantic_dim = 300
        elif self.addition_model == "clip":
            self.semantic_dim = 512

        self.fixed_bg = False
        self.class_names = get_class_name(cfg)
        self.class_embedding = get_class_embedding(
            self.class_names, self.addition_model, include_bg=self.fixed_bg
        )
        if not self.fixed_bg:
            self.bg_feature_init = torch.randn(1, self.semantic_dim)
            self.bg_feature = nn.parameter.Parameter(
                self.bg_feature_init.clone(), requires_grad=True
            )

        self.combined2vis_proj = nn.Conv2d(
            self.semantic_dim + self.visual_dim,
            self.visual_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        # self.combined2vis_proj.apply(self._init_weight)

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
        )
        # self.student_adapter.apply(self._init_weight)

        # self.discriminator = Discriminator(self.visual_dim)
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
        )
        # self.discriminator.apply(self._init_weight)

        if len(self._SHAPE_) > 1:
            self.vis2sem_proj = nn.ModuleDict()
            for scale in self._SHAPE_:
                # if scale == "res4":
                #     continue
                self.vis2sem_proj[scale] = nn.Conv2d(
                    self._SHAPE_[scale].channels,
                    self.semantic_dim,
                    kernel_size=1,
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
            nn.init.kaiming_uniform_(
                module.weight,
                nonlinearity='relu'
            )

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

        adv_losses = {}
        real_features = None
        if self.teacher_training:
            features = {
                f: self._add_semantic_features(features[f], gt_instances)
                for f in features
            }

        if self.student_training:
            features = {f: self.student_adapter(features[f]) for f in features}
            if self.distill_on and self.training:
                teacher_features = {
                    f: self._add_semantic_features(features[f], gt_instances)
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
            images, features_de_rcnn, proposals, gt_instances, real_features
        )

        # adv_losses, detector_losses = self._merge_losses(adv_losses, detector_losses)
        return adv_losses, proposal_losses, detector_losses, results, images.image_sizes

    def _merge_losses(self, adv_losses, detector_losses, alpha=10):
        loss_kd = detector_losses.pop("loss_kd")
        adv_losses.update({"loss_adv_g": adv_losses["loss_adv_g"] + alpha * loss_kd})

        return adv_losses, detector_losses

    # def _forward_discriminator(self, features):
    #     features = self.discriminator(features)
    #     features = features.permute(0, 2, 3, 1).contiguous()
    #     logits = self.discriminator_out_layer(features)
    #     return logits

    def _forward_gan(self, real_features, fake_features):
        # logit_real = self._forward_discriminator(real_features)
        # logit_fake = self._forward_discriminator(fake_features)
        logit_real = self.discriminator(real_features)
        logit_fake = self.discriminator(fake_features)

        gen_loss = F.mse_loss(fake_features, torch.ones_like(fake_features))
        disc_loss_real = F.mse_loss(logit_real, torch.ones_like(logit_real))
        disc_loss_fake = F.mse_loss(logit_fake, torch.zeros_like(logit_fake))
        disc_loss = disc_loss_real + disc_loss_fake

        return gen_loss, disc_loss

    def _forward_student_adapter(self, x):
        out = self.student_adapter(x)
        out += x
        out = F.relu(out)
        return out

    def _expand_bbox(self, gt_box, H, W, stride, expand_rate=1.0):
        x1, y1, x2, y2 = gt_box / stride
        w, h, x_c, y_c = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
        w, h = w * expand_rate, h * expand_rate
        x1 = int(max(0, x_c - w / 2))
        y1 = int(max(0, y_c - h / 2))
        x2 = int(min(W, x_c + w / 2))
        y2 = int(min(H, y_c + h / 2))
        return x1, y1, x2, y2

    def _generate_semantic_features(self, B, H, W, gt_instances, stride, expand_rate):
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

    def _distillate_multi_scale_features(self, visual_features, gt_instances):
        losses = {}
        for scale in visual_features:
            stride = self._SHAPE_[scale].stride
            if scale == "res2":
                expand_rate = 1.0
            elif scale == "res3":
                expand_rate = 1.0
            else:  # scale == 'res4':
                # continue
                expand_rate = 1.0
            B, _, H, W = visual_features[scale].shape
            semantic_features = self._generate_semantic_features(
                B, H, W, gt_instances, stride, expand_rate
            )
            projected_visual_features = self.vis2sem_proj[scale](visual_features[scale])

            losses.update(
                {
                    f"loss_rpn_{scale}": F.mse_loss(
                        projected_visual_features, semantic_features
                    )
                }
            )

        return losses
