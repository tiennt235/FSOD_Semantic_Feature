import torch
import logging
import numpy as np
from torch import nn
from typing import Dict

from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)

from .box_head import build_box_head
from .fast_rcnn import (
    ROI_HEADS_OUTPUT_REGISTRY,
    FastRCNNOutputLayers,
    FastRCNNOutputs,
    DCFastRCNNOutputs,
    KDFastRCNNOutputs,
)

from ..meta_arch.gdl import decouple_layer, AffineLayer

from ..modules import *
from defrcn.utils.class_name import get_class_name

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, self.out_channels = self._build_res5_block(cfg)
        self.output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(cls_features)

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class SemanticRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.inference_with_gt = cfg.ADDITION.INFERENCE_WITH_GT
        self.student_adapter = MLPAdapter(
            input_size=self.out_channels,
            hidden_size=self.out_channels // 2,
            output_size=self.out_channels,
        )
        self.class_names = get_class_name(cfg)
        # self.addition = ConcatAddition(
        #     input_size=self.out_channels,
        #     output_size=self.out_channels,
        #     class_names=self.class_names
        #   )
        self.addition = AttentionAddition(
            input_size=self.out_channels,
            output_size=self.out_channels,
            class_names=self.class_names,
        )

        self.student_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )
        self.mlp_adapter = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 2),
            nn.ReLU(),
            nn.Linear(self.out_channels // 2, self.out_channels),
            nn.ReLU(),
        )

    def _get_gt_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )
        positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[
            0
        ]
        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        sampled_idxs = torch.cat([positive, negative], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_proposals(self, proposals, targets):
        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_teacher(self, feature_pooled, proposals=None):
        losses = {}
        features = feature_pooled

        gt_classes = cat([p.gt_classes for p in proposals], dim=-1)
        sem_vis_features = self.addition(feature_pooled, gt_classes)
        features = sem_vis_features

        pred_class_logits, pred_proposal_deltas = self.box_predictor(features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs, losses, features, pred_class_logits

    def forward_student(
        self, proposals, feature_pooled, teacher_features, teacher_pred_class_logits
    ):
        losses = {}

        adapted_features = self.mlp_adapter(feature_pooled)

        if self.training:
            l2_loss = F.mse_loss(adapted_features, teacher_features)
            losses.update({"loss_kd_mlp": l2_loss})

        pred_class_logits, pred_proposal_deltas = self.student_box_predictor(
            adapted_features
        )

        if self.training:
            kl_loss = nn.KLDivLoss(log_target=True)(
                F.log_softmax(pred_class_logits, dim=1),
                F.log_softmax(teacher_pred_class_logits, dim=1),
            )
            losses.update({"loss_kd_att": kl_loss})

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs, losses

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif self.inference_with_gt:
            proposals = self.label_proposals(proposals, targets)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        # if self.teacher_training or (
        #     self.student_training and self.training and self.distill_mode
        # ):
        #     teacher_ouputs, teacher_losses, teacher_features, teacher_pred_class_logits = self.forward_teacher(
        #         feature_pooled,
        #         proposals
        #     )
        # if self.student_training:
        #     student_outputs, student_losses = self.forward_student(
        #         proposals,
        #         feature_pooled,
        #         teacher_features,
        #         teacher_pred_class_logits
        #     )
        if self.training:
            (
                teacher_ouputs,
                teacher_losses,
                teacher_features,
                teacher_pred_class_logits,
            ) = self.forward_teacher(feature_pooled, proposals)
            student_outputs, student_losses = self.forward_student(
                proposals, feature_pooled, teacher_features, teacher_pred_class_logits
            )

            del feature_pooled
            del features

            losses = {}
            teacher_losses.update(
                {f"{key}_t": value for key, value in (teacher_ouputs.losses().items())}
            )
            losses.update(teacher_losses)

            student_losses.update(
                {f"{key}_s": value for key, value in (student_outputs.losses().items())}
            )
            losses.update(student_losses)

            return [], losses
        else:
            student_outputs, student_losses = self.forward_student(
                proposals,
                feature_pooled,
                None,
                None,
            )

            del feature_pooled
            del features

            pred_instances, _ = student_outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class KDRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.freeze_box_predictor = cfg.MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR
        self.inference_with_gt = cfg.MODEL.ADDITION.INFERENCE_WITH_GT
        self.teacher_training = cfg.MODEL.ADDITION.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ADDITION.STUDENT_TRAINING
        self.distill_on = cfg.MODEL.ADDITION.DISTILL_ON
        self.kd_temp = cfg.MODEL.ADDITION.KD_TEMP

        self.class_names = get_class_name(cfg)
        self.addition = ConcatAddition(
            input_size=self.out_channels,
            output_size=self.out_channels,
            class_names=self.class_names,
        )
        # self.addition = AttentionAddition(
        #     input_size=self.out_channels,
        #     output_size=self.out_channels,
        #     class_names=self.class_names,
        #     )

        if self.student_training:
            self.student_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(
                self.output_layer
            )(cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg)
            self.mlp_adapter = nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels // 2),
                nn.ReLU(),
                nn.Linear(self.out_channels // 2, self.out_channels),
                nn.ReLU(),
            )

    def _get_gt_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )
        positive = nonzero_tuple((gt_classes != -1) & (gt_classes != self.num_classes))[
            0
        ]
        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        sampled_idxs = torch.cat([positive, negative], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_proposals(self, proposals, targets):
        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_teacher(self, feature_pooled, proposals=None):
        losses = {}
        features = feature_pooled

        gt_classes = cat([p.gt_classes for p in proposals], dim=-1)
        sem_vis_features = self.addition(feature_pooled, gt_classes)
        features = sem_vis_features

        pred_class_logits, pred_proposal_deltas = self.box_predictor(features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs, losses, features, pred_class_logits

    def forward_student(
        self, proposals, feature_pooled, teacher_features, teacher_pred_class_logits
    ):
        losses = {}

        adapted_features = self.mlp_adapter(feature_pooled)

        if self.training:
            l2_loss = F.mse_loss(adapted_features, teacher_features)
            losses.update({"loss_kd_mlp": l2_loss})

        pred_class_logits, pred_proposal_deltas = self.student_box_predictor(
            adapted_features
        )

        if self.training:
            temp = self.kl_temp
            kl_loss = (
                nn.KLDivLoss(log_target=True)(
                    F.log_softmax(pred_class_logits, dim=1) / temp,
                    F.log_softmax(teacher_pred_class_logits, dim=1) / temp,
                )
                * temp
                * temp
            )
            losses.update({"loss_kd_addition": kl_loss})

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs, losses

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif self.inference_with_gt:
            proposals = self.label_proposals(proposals, targets)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        teacher_features = None
        teacher_pred_class_logits = None
        if self.teacher_training or (
            self.student_training and self.training and self.distill_on
        ):
            (
                teacher_ouputs,
                teacher_losses,
                teacher_features,
                teacher_pred_class_logits,
            ) = self.forward_teacher(feature_pooled, proposals)

        if self.student_training:
            student_outputs, student_losses = self.forward_student(
                proposals, feature_pooled, teacher_features, teacher_pred_class_logits
            )

        del feature_pooled

        if self.training:
            del features

            losses = {}
            if not self.freeze_box_predictor:
                teacher_losses.update(
                    {
                        f"{key}_t": value
                        for key, value in (teacher_ouputs.losses().items())
                    }
                )
                losses.update(teacher_losses)

            if self.student_training:
                student_losses.update(
                    {
                        f"{key}_s": value
                        for key, value in (student_outputs.losses().items())
                    }
                )
                losses.update(student_losses)

            return [], losses

        else:
            if self.teacher_training:
                outputs = teacher_ouputs
            if self.student_training:
                outputs = student_outputs

            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )

            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class KDFMRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
<<<<<<< HEAD

        self.teacher_training = cfg.MODEL.ADDITION.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ADDITION.STUDENT_TRAINING
        self.distill_on = cfg.MODEL.ADDITION.DISTILL_ON
        self.kd_temp = cfg.MODEL.ADDITION.KD_TEMP
            
=======
        self.kd_temp = cfg.MODEL.ADDITION.KD_TEMP
        
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
    def forward(self, images, features, proposals, targets=None, real_features=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # elif self.inference_with_gt:
        #     proposals = self.label_proposals(proposals, targets)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )

        del feature_pooled
        
<<<<<<< HEAD
=======
        # teacher_pred_class_logits = None
        # teacher_pred_proposal_deltas = None
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        real_pred_class_logits = None
        real_pred_proposal_deltas = None
        teacher_pred_class_logits = None
        teacher_pred_proposal_deltas = None
        
        if self.training:
<<<<<<< HEAD
            if self.distill_on:
                real_box_features = self._shared_roi_transform(
                    [real_features[f] for f in self.in_features], proposal_boxes
                )
                real_feature_pooled = real_box_features.mean(dim=[2, 3])  # pooled to 1x1

                real_pred_class_logits, real_pred_proposal_deltas = self.box_predictor(
                    real_feature_pooled
                )
                
                # teacher_pred_class_logits, teacher_pred_proposal_deltas = self.box_predictor(
                #     real_feature_pooled
                # )
                
                del real_features
                del real_feature_pooled
=======
            real_box_features = self._shared_roi_transform(
                [real_features[f] for f in self.in_features], proposal_boxes
            )
            real_feature_pooled = real_box_features.mean(dim=[2, 3])  # pooled to 1x1

            real_pred_class_logits, real_pred_proposal_deltas = self.box_predictor(
                real_feature_pooled
            )
            # teacher_pred_class_logits, teacher_pred_proposal_deltas = self.box_predictor(
            #     real_feature_pooled
            # )

            # real_pred_class_logits, real_pred_proposal_deltas = self.teacher_box_predictor(
            #     real_feature_pooled
            # )
            
            del real_features
            del real_feature_pooled
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
        
        outputs = KDFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.kd_temp,
            real_pred_class_logits,
            real_pred_proposal_deltas,
            teacher_pred_class_logits,
            teacher_pred_proposal_deltas,
            self.teacher_training
        )


        if self.training:
            del features
            losses = outputs.losses()
            return [], losses

        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )

            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class DCRes5ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def forward(self, images, features, proposals, targets=None, fs_class=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = DCFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            fs_class
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            # pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}