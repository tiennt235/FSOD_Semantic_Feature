
from .roi_heads import *


@ROI_HEADS_REGISTRY.register()
class SuperRes5ROIHeads2(Res5ROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.use_mem = cfg.MODEL.ROI_HEADS.USE_MEMORY
        self.use_ot = cfg.MODEL.ROI_HEADS.USE_OT
        self.is_freeze_mpl = cfg.MODEL.ROI_HEADS.FREEZE_MPL
        self.use_background = True
        self.repeat_time = cfg.MODEL.ROI_HEADS.REPEATED_TIME
        self.factors = cfg.MODEL.ROI_HEADS.FACTORS
        self.capacity = cfg.MODEL.ROI_HEADS.MEM_CAPACITY
        self.use_bbox = cfg.MODEL.ROI_HEADS.USE_BBX
        self.meta_loss_weight = cfg.MODEL.META_LOSS_WEIGHT
        self.syn_loss_weight = cfg.MODEL.SYN_LOSS_WEIGHT
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.novel_tuning = True
        self.save_dir = cfg.OUTPUT_DIR
        self.init_super_cls2(cfg)
        self.student_l2_loss = cfg.MODEL.ROI_HEADS.L2
        self.student_kl_loss = cfg.MODEL.ROI_HEADS.KL
        self.student_kl_temp = cfg.MODEL.ROI_HEADS.KL_TEMP

        self.__init_LV_model__(self.out_channels, cfg)

        # super_num_class = self.num_group*self.num_k
        # is_super = False
        # num_class = super_num_class if is_super else self.num_classes
        num_class = self.num_classes

        self.stu_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.tracker_copy_weight = False

    def init_super_cls2(self, cfg):
        self.get_class(cfg)

        # super_class = cfg.MODEL.ROI_HEADS.PSEUDO_CLASS_VOC
        # super_class = {key: val for [key, val] in super_class}
        # super_class = torch.tensor([super_class[i] for i in self.classes])
        # bg_ = torch.ones(1, super_class.shape[1]) * (self.num_k-1)
        # super_class = torch.cat([super_class, bg_], dim=0)
        # print('super_class', super_class)
        # self.super_class = super_class.to(int).cuda()
        # assert 0
        return

    def get_class(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        is_coco = True if 'coco' in dataset_name else False

        if is_coco:
            data = {i['name']: i for i in cfg.SUPER_CLASS}
        else:
            data = {i['voc_name']: i for i in cfg.SUPER_CLASS}

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                # metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        super_class = [data[name]['super_id'] for name in self.classes]

        unique_super_cls = set(super_class)
        self.num_super_cls = len(unique_super_cls)

        cls_id_to_contiguous_id = {k: i for i,
                                   k in enumerate(unique_super_cls)}
        self.mapper = torch.tensor(
            [cls_id_to_contiguous_id[i] for i in super_class])

        # self.super2class = np.where()

        self.cls2fg = self.mapper.clone()

        for i in self.mapper.unique():
            index = torch.where(self.mapper == i)[0]
            self.cls2fg[index] = torch.arange(len(index))

        print(self.mapper)
        # assert 0
        return

    def init_super_cls(self, cfg):
        self.get_class(cfg)
        self.num_k = cfg.MODEL.K_CLASS + 1  # add bg
        self.num_group = cfg.MODEL.ROI_HEADS.NUM_GROUP_SUPER

        super_class = cfg.MODEL.ROI_HEADS.PSEUDO_CLASS_VOC
        super_class = {key: val for [key, val] in super_class}
        super_class = torch.tensor([super_class[i] for i in self.classes])
        bg_ = torch.ones(1, super_class.shape[1]) * (self.num_k-1)
        super_class = torch.cat([super_class, bg_], dim=0)
        print('super_class', super_class)
        self.super_class = super_class.to(int).cuda()

    def get_class1(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention1(input_size, cfg=cfg, is_multi=False)
        if self.student_training:
            self.mlp_adapter = MLP(input_size, widening_factor=2)
            self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass

    def forward_with_text_attention(self, fg_features, num_preds_per_image=None, gt_classes=None):
        # with torch.no_grad():
        #     roi_gt, roi_cls = self.get_roi_with_gt2(
        #         [features[f] for f in self.in_features], targets)
        # ot_feat_input = torch.cat(roi_gt, dim=0).unsqueeze(0)
        # ot_cls_input = torch.cat(roi_cls, dim=0).unsqueeze(0)

        feat = self.attention(
            fg_features, gt_classes, num_preds_per_image)
        # att = torch.cat(sim2stext, gim2gtext)
        # att = self.pro_att(att).softmax(dim=-1)
        # fg_features = self.atten_bb(fg_features, None)[0]
        # print('fg_features: ', fg_features.shape)
        # assert 0
        # att_ft = sim2stext+gim2gtext
        return feat

    def forward_adapter(self, fg_features, teacher_features):

        # feat = self.attention.forward_wo_label(fg_features)
        feat = self.mlp_adapter(fg_features)
        loss = {}
        alpha = 1.0
        margin = 0.2
        def norm_x(x): return F.normalize(x)

        def loss_cosine(a, b):
            return 1 - torch.einsum(
                'b i, b i -> b', norm_x(a), norm_x(b))

        if self.training and self.distill_mode and self.student_l2_loss:
            # l = ((feat - teacher_features)**2).mean()*0

            # l = loss_cosine(feat, teacher_features).mean()*alpha
            # l = (l - margin).clamp(min=0.2)
            # l[torch.where(l < margin)] = 0
            # l = l.mean()*alpha
            # print('feat', feat)
            # print('teacher_features', teacher_features)
            l = F.mse_loss(feat, teacher_features)*alpha

            l = F.cosine_similarity(feat, teacher_features)
            l = F.cosine_embedding_loss(feat, teacher_features)

            # l = l/feat.shape[0]  # mean according batch size
            loss = {'loss_student_feat': l}

        return feat, loss

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

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

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
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_teacher(self, feature_pooled, proposals, test_with_gt=True):

        gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        num_preds_per_image = [len(p) for p in proposals]
        loss_att, output_att = self.forward_with_text_attention(
            feature_pooled, gt_classes=gt_classes, num_preds_per_image=num_preds_per_image)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward_student(self, feature_pooled, proposals, teacher_output):

        teacher_features = teacher_output.get('sim2stext', None)

        att_feature, loss = self.forward_adapter(
            feature_pooled, teacher_features=teacher_features)

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        if self.student_training and self.training and self.distill_mode and self.student_kl_loss:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': self.student_kl_temp,  # 1, 5, 10, 15
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward(self, images, features, proposals, targets=None):
        del images
        test_with_gt = True if (not self.training) and targets else False
        # print('test_with_gt:', test_with_gt)
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        t_output = {}
        if self.teacher_training:
            t_output, t_loss = self.forward_teacher(
                feature_pooled, proposals, test_with_gt)
            t_loss = {key+'_t': val for key, val in t_loss.items()}

            # t_loss = {key+'_t' : val for key, val in t_loss.items()}
            teacher_outputs = FastRCNNOutputs(
                self.box2box_transform,
                t_output['pred_logits'],
                t_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.student_training:
            s_output, s_loss = self.forward_student(
                feature_pooled, proposals, t_output)

            student_outputs = FastRCNNOutputs(
                self.box2box_transform,
                s_output['pred_logits'],
                s_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            losses = {}
            if self.teacher_training:
                loss = teacher_outputs.losses()
                loss = {key+'_t': val for key, val in loss.items()}

                losses.update(loss)
                losses.update(t_loss)
            if self.student_training:
                losses.update(student_outputs.losses())
                losses.update(s_loss)

            return [], losses
        else:
            if self.teacher_training:
                pred_instances, _ = teacher_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            if self.student_training:
                pred_instances, _ = student_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class SuperRes5ROIHeads3(Res5ROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.use_mem = cfg.MODEL.ROI_HEADS.USE_MEMORY
        self.use_ot = cfg.MODEL.ROI_HEADS.USE_OT
        self.is_freeze_mpl = cfg.MODEL.ROI_HEADS.FREEZE_MPL
        self.use_background = True
        self.repeat_time = cfg.MODEL.ROI_HEADS.REPEATED_TIME
        self.factors = cfg.MODEL.ROI_HEADS.FACTORS
        self.capacity = cfg.MODEL.ROI_HEADS.MEM_CAPACITY
        self.use_bbox = cfg.MODEL.ROI_HEADS.USE_BBX
        self.meta_loss_weight = cfg.MODEL.META_LOSS_WEIGHT
        self.syn_loss_weight = cfg.MODEL.SYN_LOSS_WEIGHT
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.novel_tuning = True
        self.save_dir = cfg.OUTPUT_DIR
        self.get_class(cfg)
        self.__init_LV_model__(self.out_channels, cfg)
        self.student_l2_loss = cfg.MODEL.ROI_HEADS.L2
        self.student_kl_loss = cfg.MODEL.ROI_HEADS.KL
        self.student_kl_temp = cfg.MODEL.ROI_HEADS.KL_TEMP

        self.super_training = cfg.MODEL.ROI_HEADS.SUPER_TRAINING

        # super_num_class = self.num_group*self.num_k
        # is_super = False
        # num_class = super_num_class if is_super else self.num_classes
        num_class = self.num_classes

        self.stu_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.tracker_copy_weight = False
        # self.affine_attention = AffineLayer(
        #     num_channels=self.out_channels, bias=True)

    def get_class(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        is_coco = True if 'coco' in dataset_name else False

        if is_coco:
            data = {i['name']: i for i in cfg.SUPER_CLASS}
        else:
            data = {i['voc_name']: i for i in cfg.SUPER_CLASS}

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                # metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        super_class = [data[name]['super_id'] for name in self.classes]

        unique_super_cls = set(super_class)
        self.num_super_cls = len(unique_super_cls)

        cls_id_to_contiguous_id = {k: i for i,
                                   k in enumerate(unique_super_cls)}
        self.mapper = [cls_id_to_contiguous_id[i] for i in super_class]
        self.mapper.append(len(set(self.mapper)))

        self.mapper = torch.tensor(self.mapper)

        # self.super2class = np.where()

        self.cls2fg = self.mapper.clone()

        for i in self.mapper.unique():
            index = torch.where(self.mapper == i)[0]
            self.cls2fg[index] = torch.arange(len(index))

        self.cls2fg = self.cls2fg.cuda().long()
        self.mapper = self.mapper.cuda().long()

        self.super_extractor = torch.nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=True),
            nn.ReLU(),
            # nn.Linear(self.out_channels//4, self.out_channels, bias=True),
            # nn.ReLU(),
        )

        # self.fine_grained_extractor = torch.nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(self.out_channels, self.out_channels, bias=True),
        #     nn.ReLU(),
        # )
        self.num_super_cls = self.mapper.unique().shape[0]
        self.super_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_super_cls, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )

        # self.fine_graind_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
        #     cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg,
        #     # num_super_classes=super_num_class
        # )

        print(self.mapper)
        # assert 0
        return

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention(
            input_size, cfg=cfg, is_multi=True, num_super_cls=self.num_super_cls, dropout=cfg.MODEL.ROI_HEADS.DROPOUT_ATTENTION)
        # input_size, cfg=cfg, is_multi=False, num_super_cls=self.num_super_cls)
        if self.student_training:
            self.mlp_adapter = MLP(input_size, widening_factor=2)
            self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass

    def forward_with_text_attention(self, fg_features, num_preds_per_image=None, gt_classes=None):
        # with torch.no_grad():
        #     roi_gt, roi_cls = self.get_roi_with_gt2(
        #         [features[f] for f in self.in_features], targets)
        # ot_feat_input = torch.cat(roi_gt, dim=0).unsqueeze(0)
        # ot_cls_input = torch.cat(roi_cls, dim=0).unsqueeze(0)

        feat = self.attention(
            fg_features, gt_classes, num_preds_per_image)
        # att = torch.cat(sim2stext, gim2gtext)
        # att = self.pro_att(att).softmax(dim=-1)
        # fg_features = self.atten_bb(fg_features, None)[0]
        # print('fg_features: ', fg_features.shape)
        # assert 0
        # att_ft = sim2stext+gim2gtext
        return feat

    def forward_adapter(self, fg_features, teacher_features):

        # feat = self.attention.forward_wo_label(fg_features)
        feat = self.mlp_adapter(fg_features)
        loss = {}
        alpha = 1.0
        margin = 0.2
        def norm_x(x): return F.normalize(x)

        def loss_cosine(a, b):
            return 1 - torch.einsum(
                'b i, b i -> b', norm_x(a), norm_x(b))

        if self.training and self.distill_mode and self.student_l2_loss:
            # l = ((feat - teacher_features)**2).mean()*0

            # l = loss_cosine(feat, teacher_features).mean()*alpha
            # l = (l - margin).clamp(min=0.2)
            # l[torch.where(l < margin)] = 0
            # l = l.mean()*alpha
            l = F.mse_loss(feat, teacher_features)*alpha

            # # l = l/feat.shape[0]  # mean according batch size
            loss = {'loss_student_feat': l}
            # pass

        return feat, loss

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

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

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
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_visual_super(self, feature_pooled, proposals):
        scale = 0.1
        scale = 0.1

        if not self.super_training:
            scale = 0.0
        # scale = 1
        super_feat = self.super_extractor(
            decouple_layer(feature_pooled, scale))

        # fg_feat = self.fine_grained_extractor(feature_pooled)
        losses = {'vis_super_deconv_loss': Deconv_loss(super_feat)}
        super_logits, super_bbox = self.super_predictor(super_feat)

        if not self.super_training:
            output = {
                'vis_sp_feat': super_feat,
                'vis_fg_feat': feature_pooled,
                'super_logits': super_logits,
            }

            return output, {}
        # fg_logits, fg_bbox = self.fine_graind_predictor(fg_feat)
        fg_logits = None
        fg_feat = None
        output_super = SuperClassOutputs(
            box2box_transform=self.box2box_transform,
            pred_class_logits=[super_logits, fg_logits],
            pred_proposal_deltas=super_bbox,
            proposals=proposals,
            smooth_l1_beta=self.smooth_l1_beta,
            model_method='default',
            eval_method='none',
            eval_gt_classes=None,
            eval_ways=1,
            cosine_scale=-1.0,
            mapper=self.mapper,
            cls2fg=self.cls2fg,
            use_super_cls_acti=True,
            is_multi_super_cls=False,
            use_fine_grained_cls_acti=True,
            use_margin_loss=False,
        )

        fg_feat = feature_pooled if isinstance(
            fg_feat, type(None)) else fg_feat

        super_feat
        output = {
            'vis_sp_feat': super_feat,
            'vis_fg_feat': fg_feat,
            'super_logits': super_logits,

        }
        if self.training:
            # print(losses)
            losses.update(output_super.losses())
            return output, losses
        else:
            return output, {}

    def forward_teacher(self, inputs, proposals, test_with_gt=True):
        feature_pooled = inputs['feature_pooled']
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        fg_ind = gt_classes != len(self.mapper)
        gt_super_class = self.mapper[gt_classes[fg_ind]]

        num_preds_per_image = [len(p) for p in proposals]
        # loss_att, output_att = self.forward_with_text_attention(
        #     inputs, gt_classes=gt_classes, num_preds_per_image=num_preds_per_image)
        loss_att, output_att = self.forward_with_text_attention(
            inputs, gt_classes=[gt_classes, [fg_ind, gt_super_class]], num_preds_per_image=num_preds_per_image)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward_student(self, feature_pooled, proposals, teacher_output):

        teacher_features = teacher_output.get('sim2stext', None)
        att_feature = teacher_output.get('distill_feat', None)
        att_feature, loss = self.forward_adapter(
            feature_pooled, teacher_features=teacher_features)

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        # if self.student_training and self.training and self.distill_mode and self.student_kl_loss:
        if self.training and self.distill_mode and self.student_kl_loss:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': self.student_kl_temp,
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward_student2(self, feature_pooled, proposals, teacher_output):

        att_feature = teacher_output.get('distill_feat', None)
        loss = {}

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        # if self.student_training and self.training and self.distill_mode:
        if self.training and self.distill_mode:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': 1,
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward(self, images, features, proposals, targets=None):
        del images
        test_with_gt = True if (not self.training) and targets else False
        # print('test_with_gt:', test_with_gt)
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        super_loss = {}

        output_super, super_loss = self.forward_visual_super(
            feature_pooled, proposals)

        output_super['feature_pooled'] = feature_pooled

        # print(super_loss)
        # assert 0
        t_output = {}
        if self.teacher_training:
            t_output, t_loss = self.forward_teacher(
                output_super, proposals, test_with_gt)
            t_loss = {'tea_'+key: val for key, val in t_loss.items()}

            teacher_outputs = FastRCNNOutputs(
                self.box2box_transform,
                t_output['pred_logits'],
                t_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

            # teacher_outputs = SuperClassOutputs2(
            #     box2box_transform=self.box2box_transform,
            #     pred_class_logits=[
            #         output_super['super_logits'], t_output['pred_logits']],
            #     pred_proposal_deltas=t_output['pred_bbox'],
            #     proposals=proposals,
            #     smooth_l1_beta=self.smooth_l1_beta,
            #     model_method='default',
            #     eval_method='none',
            #     eval_gt_classes=None,
            #     eval_ways=1,
            #     cosine_scale=-1.0,
            #     mapper=self.mapper,
            #     cls2fg=self.cls2fg,
            #     use_super_cls_acti=True,
            #     is_multi_super_cls=False,
            #     use_fine_grained_cls_acti=True,
            #     use_margin_loss=False,
            # )

        if self.student_training:
            # s_output, s_loss = self.forward_student(
            # s_output, s_loss = self.forward_student2(
            s_output, s_loss = self.forward_student(
                feature_pooled, proposals, t_output)
            s_loss = {'stu_'+key: val for key, val in s_loss.items()}

            student_outputs = FastRCNNOutputs(
                self.box2box_transform,
                s_output['pred_logits'],
                s_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            losses = {}
            losses.update(super_loss)
            if self.teacher_training:
                # loss = teacher_outputs.losses()
                # loss = {'tea_'+key: val for key,
                #         val in teacher_outputs.losses().items()}

                # loss = {key+'_t': val for key, val in loss.items()}

                losses.update({'tea_'+key: val for key,
                              val in teacher_outputs.losses().items()})
                losses.update(t_loss)
            if self.student_training:
                # s_loss = student_outputs.losses()
                # s_loss = {'stu_'+key: val for key, val in s_loss.items()}
                losses.update({'stu_'+key: val for key,
                              val in student_outputs.losses().items()})

                losses.update(s_loss)

            return [], losses
        else:
            if self.teacher_training:
                pred_instances, _ = teacher_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            if self.student_training:
                pred_instances, _ = student_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class SuperRes5ROIHeads4_feat(Res5ROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.use_mem = cfg.MODEL.ROI_HEADS.USE_MEMORY
        self.use_ot = cfg.MODEL.ROI_HEADS.USE_OT
        self.is_freeze_mpl = cfg.MODEL.ROI_HEADS.FREEZE_MPL
        self.use_background = True
        self.repeat_time = cfg.MODEL.ROI_HEADS.REPEATED_TIME
        self.factors = cfg.MODEL.ROI_HEADS.FACTORS
        self.capacity = cfg.MODEL.ROI_HEADS.MEM_CAPACITY
        self.use_bbox = cfg.MODEL.ROI_HEADS.USE_BBX
        self.meta_loss_weight = cfg.MODEL.META_LOSS_WEIGHT
        self.syn_loss_weight = cfg.MODEL.SYN_LOSS_WEIGHT
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.novel_tuning = True
        self.save_dir = cfg.OUTPUT_DIR
        self.get_class(cfg)
        self.__init_LV_model__(self.out_channels, cfg)
        self.student_l2_loss = cfg.MODEL.ROI_HEADS.L2
        self.student_kl_loss = cfg.MODEL.ROI_HEADS.KL
        self.student_kl_temp = cfg.MODEL.ROI_HEADS.KL_TEMP

        self.super_training = cfg.MODEL.ROI_HEADS.SUPER_TRAINING

        # super_num_class = self.num_group*self.num_k
        # is_super = False
        # num_class = super_num_class if is_super else self.num_classes
        num_class = self.num_classes

        self.stu_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.tracker_copy_weight = False
        # self.affine_attention = AffineLayer(
        #     num_channels=self.out_channels, bias=True)

    def get_class(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        is_coco = True if 'coco' in dataset_name else False

        if is_coco:
            data = {i['name']: i for i in cfg.SUPER_CLASS}
        else:
            data = {i['voc_name']: i for i in cfg.SUPER_CLASS}

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                # metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        super_class = [data[name]['super_id'] for name in self.classes]

        unique_super_cls = set(super_class)
        self.num_super_cls = len(unique_super_cls)

        cls_id_to_contiguous_id = {k: i for i,
                                   k in enumerate(unique_super_cls)}
        self.mapper = [cls_id_to_contiguous_id[i] for i in super_class]
        self.mapper.append(len(set(self.mapper)))

        self.mapper = torch.tensor(self.mapper)

        # self.super2class = np.where()

        self.cls2fg = self.mapper.clone()

        for i in self.mapper.unique():
            index = torch.where(self.mapper == i)[0]
            self.cls2fg[index] = torch.arange(len(index))

        self.cls2fg = self.cls2fg.cuda().long()
        self.mapper = self.mapper.cuda().long()

        self.super_extractor = torch.nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=True),
            nn.ReLU(),
            # nn.Linear(self.out_channels//4, self.out_channels, bias=True),
            # nn.ReLU(),
        )

        # self.fine_grained_extractor = torch.nn.Sequential(
        #     nn.Linear(self.out_channels, self.out_channels, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(self.out_channels, self.out_channels, bias=True),
        #     nn.ReLU(),
        # )
        self.num_super_cls = self.mapper.unique().shape[0]
        self.super_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, self.num_super_cls, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )

        # self.fine_graind_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
        #     cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg,
        #     # num_super_classes=super_num_class
        # )

        print(self.mapper)
        # assert 0
        return

    def __init_LV_model__(self, input_size, cfg):
        # return
        # print('self.pooler_scales', self.pooler_scales)
        # assert 0
        self.attention = LV_attention_feat(
            input_size, cfg=cfg, is_multi=True, num_super_cls=self.num_super_cls, pooler_scales=self.pooler_scales[0], dropout=cfg.MODEL.ROI_HEADS.DROPOUT_ATTENTION)
        # input_size, cfg=cfg, is_multi=False, num_super_cls=self.num_super_cls)
        if self.student_training:
            self.mlp_adapter = MLP(input_size, widening_factor=2)
            self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
                nn.Linear(input_size, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    # def forward_with_text_attention(self, fg_features, num_preds_per_image=None, gt_classes=None):
    #     # with torch.no_grad():
    #     #     roi_gt, roi_cls = self.get_roi_with_gt2(
    #     #         [features[f] for f in self.in_features], targets)
    #     # ot_feat_input = torch.cat(roi_gt, dim=0).unsqueeze(0)
    #     # ot_cls_input = torch.cat(roi_cls, dim=0).unsqueeze(0)

    #     feat =
    #     # att = torch.cat(sim2stext, gim2gtext)
    #     # att = self.pro_att(att).softmax(dim=-1)
    #     # fg_features = self.atten_bb(fg_features, None)[0]
    #     # print('fg_features: ', fg_features.shape)
    #     # assert 0
    #     # att_ft = sim2stext+gim2gtext
    #     return feat

    def forward_adapter(self, fg_features, teacher_features):

        # feat = self.attention.forward_wo_label(fg_features)
        feat = self.mlp_adapter(fg_features)
        loss = {}
        alpha = 1.0
        margin = 0.2
        def norm_x(x): return F.normalize(x)

        def loss_cosine(a, b):
            return 1 - torch.einsum(
                'b i, b i -> b', norm_x(a), norm_x(b))

        if self.training and self.distill_mode and self.student_l2_loss:
            # l = ((feat - teacher_features)**2).mean()*0

            # l = loss_cosine(feat, teacher_features).mean()*alpha
            # l = (l - margin).clamp(min=0.2)
            # l[torch.where(l < margin)] = 0
            # l = l.mean()*alpha
            l = F.mse_loss(feat, teacher_features)*alpha

            # # l = l/feat.shape[0]  # mean according batch size
            loss = {'loss_student_feat': l}
            # pass

        return feat, loss

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

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

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
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_visual_super(self, feature_pooled, proposals):
        scale = 0.1
        scale = 0.1

        if not self.super_training:
            scale = 0.0
        # scale = 1
        super_feat = self.super_extractor(
            decouple_layer(feature_pooled, scale))

        # fg_feat = self.fine_grained_extractor(feature_pooled)
        losses = {'vis_super_deconv_loss': Deconv_loss(super_feat)}
        super_logits, super_bbox = self.super_predictor(super_feat)

        if not self.super_training:
            output = {
                'vis_sp_feat': super_feat,
                'vis_fg_feat': feature_pooled,
                'super_logits': super_logits,
            }

            return output, {}
        # fg_logits, fg_bbox = self.fine_graind_predictor(fg_feat)
        fg_logits = None
        fg_feat = None
        output_super = SuperClassOutputs(
            box2box_transform=self.box2box_transform,
            pred_class_logits=[super_logits, fg_logits],
            pred_proposal_deltas=super_bbox,
            proposals=proposals,
            smooth_l1_beta=self.smooth_l1_beta,
            model_method='default',
            eval_method='none',
            eval_gt_classes=None,
            eval_ways=1,
            cosine_scale=-1.0,
            mapper=self.mapper,
            cls2fg=self.cls2fg,
            use_super_cls_acti=True,
            is_multi_super_cls=False,
            use_fine_grained_cls_acti=True,
            use_margin_loss=False,
        )

        fg_feat = feature_pooled if isinstance(
            fg_feat, type(None)) else fg_feat

        super_feat
        output = {
            'vis_sp_feat': super_feat,
            'vis_fg_feat': fg_feat,
            'super_logits': super_logits,

        }
        if self.training:
            # print(losses)
            losses.update(output_super.losses())
            return output, losses
        else:
            return output, {}

    def forward_teacher(self, inputs, proposals, test_with_gt=True):
        feature_pooled = inputs['feature_pooled']
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        fg_ind = gt_classes != len(self.mapper)
        gt_super_class = self.mapper[gt_classes[fg_ind]]

        num_preds_per_image = [len(p) for p in proposals]
        # loss_att, output_att = self.forward_with_text_attention(
        #     inputs, gt_classes=gt_classes, num_preds_per_image=num_preds_per_image)
        loss_att, output_att = self.attention(inputs, gt_classes=[gt_classes, [
                                              fg_ind, gt_super_class]], gt_instances=proposals)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward_student(self, feature_pooled, proposals, teacher_output):

        teacher_features = teacher_output.get('sim2stext', None)
        att_feature = teacher_output.get('distill_feat', None)
        att_feature, loss = self.forward_adapter(
            feature_pooled, teacher_features=teacher_features)

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        # if self.student_training and self.training and self.distill_mode and self.student_kl_loss:
        if self.training and self.distill_mode and self.student_kl_loss:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': self.student_kl_temp,
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward_student2(self, feature_pooled, proposals, teacher_output):

        att_feature = teacher_output.get('distill_feat', None)
        loss = {}

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        # if self.student_training and self.training and self.distill_mode:
        if self.training and self.distill_mode:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': 1,
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward(self, images, features, proposals, targets=None):
        del images
        test_with_gt = True if (not self.training) and targets else False
        # print('test_with_gt:', test_with_gt)
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        super_loss = {}

        output_super, super_loss = self.forward_visual_super(
            feature_pooled, proposals)

        output_super['feature_pooled'] = box_features

        # print(super_loss)
        # assert 0
        t_output = {}
        if self.teacher_training:
            t_output, t_loss = self.forward_teacher(
                output_super, proposals, test_with_gt)
            t_loss = {'tea_'+key: val for key, val in t_loss.items()}

            teacher_outputs = FastRCNNOutputs(
                self.box2box_transform,
                t_output['pred_logits'],
                t_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

            # teacher_outputs = SuperClassOutputs2(
            #     box2box_transform=self.box2box_transform,
            #     pred_class_logits=[
            #         output_super['super_logits'], t_output['pred_logits']],
            #     pred_proposal_deltas=t_output['pred_bbox'],
            #     proposals=proposals,
            #     smooth_l1_beta=self.smooth_l1_beta,
            #     model_method='default',
            #     eval_method='none',
            #     eval_gt_classes=None,
            #     eval_ways=1,
            #     cosine_scale=-1.0,
            #     mapper=self.mapper,
            #     cls2fg=self.cls2fg,
            #     use_super_cls_acti=True,
            #     is_multi_super_cls=False,
            #     use_fine_grained_cls_acti=True,
            #     use_margin_loss=False,
            # )

        if self.student_training:
            # s_output, s_loss = self.forward_student(
            # s_output, s_loss = self.forward_student2(
            s_output, s_loss = self.forward_student(
                feature_pooled, proposals, t_output)
            s_loss = {'stu_'+key: val for key, val in s_loss.items()}

            student_outputs = FastRCNNOutputs(
                self.box2box_transform,
                s_output['pred_logits'],
                s_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            losses = {}
            losses.update(super_loss)
            if self.teacher_training:
                # loss = teacher_outputs.losses()
                # loss = {'tea_'+key: val for key,
                #         val in teacher_outputs.losses().items()}

                # loss = {key+'_t': val for key, val in loss.items()}

                losses.update({'tea_'+key: val for key,
                              val in teacher_outputs.losses().items()})
                losses.update(t_loss)
            if self.student_training:
                # s_loss = student_outputs.losses()
                # s_loss = {'stu_'+key: val for key, val in s_loss.items()}
                losses.update({'stu_'+key: val for key,
                              val in student_outputs.losses().items()})

                losses.update(s_loss)

            return [], losses
        else:
            if self.teacher_training:
                pred_instances, _ = teacher_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            if self.student_training:
                pred_instances, _ = student_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class SuperRes5ROIHeads2_clone(Res5ROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.use_mem = cfg.MODEL.ROI_HEADS.USE_MEMORY
        self.use_ot = cfg.MODEL.ROI_HEADS.USE_OT
        self.is_freeze_mpl = cfg.MODEL.ROI_HEADS.FREEZE_MPL
        self.use_background = True
        self.repeat_time = cfg.MODEL.ROI_HEADS.REPEATED_TIME
        self.factors = cfg.MODEL.ROI_HEADS.FACTORS
        self.capacity = cfg.MODEL.ROI_HEADS.MEM_CAPACITY
        self.use_bbox = cfg.MODEL.ROI_HEADS.USE_BBX
        self.meta_loss_weight = cfg.MODEL.META_LOSS_WEIGHT
        self.syn_loss_weight = cfg.MODEL.SYN_LOSS_WEIGHT
        self.teacher_training = cfg.MODEL.ROI_HEADS.TEACHER_TRAINING
        self.student_training = cfg.MODEL.ROI_HEADS.STUDENT_TRAINING
        self.distill_mode = cfg.MODEL.ROI_HEADS.DISTILLATE
        self.novel_tuning = True
        self.save_dir = cfg.OUTPUT_DIR
        self.init_super_cls(cfg)

        self.__init_LV_model__(self.out_channels, cfg)

        super_num_class = self.num_group*self.num_k
        is_super = False
        num_class = super_num_class if is_super else self.num_classes

        self.stu_box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer)(
            cfg, self.out_channels, num_class, self.cls_agnostic_bbox_reg,
            # num_super_classes=super_num_class
        )
        self.tracker_copy_weight = False

    def init_super_cls2(self, cfg):
        self.get_class(cfg)

        # super_class = cfg.MODEL.ROI_HEADS.PSEUDO_CLASS_VOC
        # super_class = {key: val for [key, val] in super_class}
        # super_class = torch.tensor([super_class[i] for i in self.classes])
        # bg_ = torch.ones(1, super_class.shape[1]) * (self.num_k-1)
        # super_class = torch.cat([super_class, bg_], dim=0)
        # print('super_class', super_class)
        # self.super_class = super_class.to(int).cuda()
        return

    def get_class(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        is_coco = True if 'coco' in dataset_name else False

        if is_coco:
            data = {i['name']: i for i in cfg.SUPER_CLASS}
        else:
            data = {i['voc_name']: i for i in cfg.SUPER_CLASS}

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                # metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

        self.super_class = torch.tensor(
            [data[name]['super_id'] for name in self.classes])

        # self.super2class = np.where()

        # print(self.super_class)
        # assert 0
        return

    def init_super_cls(self, cfg):
        self.get_class(cfg)
        self.num_k = cfg.MODEL.K_CLASS + 1  # add bg
        self.num_group = cfg.MODEL.ROI_HEADS.NUM_GROUP_SUPER

        super_class = cfg.MODEL.ROI_HEADS.PSEUDO_CLASS_VOC
        super_class = {key: val for [key, val] in super_class}
        super_class = torch.tensor([super_class[i] for i in self.classes])
        bg_ = torch.ones(1, super_class.shape[1]) * (self.num_k-1)
        super_class = torch.cat([super_class, bg_], dim=0)
        print('super_class', super_class)
        self.super_class = super_class.to(int).cuda()

    def get_class1(self, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]

        metadata_dict = MetadataCatalog.get(dataset_name)
        is_novel = True if 'shot' in dataset_name else False

        if is_novel:
            if 'all' in dataset_name:
                self.classes = metadata_dict.thing_classes
            else:
                self.classes = metadata_dict.novel_classes
                metadata_dict.novel_dataset_id_to_contiguous_id
        else:
            self.classes = metadata_dict.base_classes

    def __init_LV_model__(self, input_size, cfg):
        # return
        self.attention = LV_attention(input_size, cfg=cfg, is_multi=False)
        if self.student_training:
            self.mlp_adapter = MLP(input_size, widening_factor=2)
            self.mlp_adapter = Adaptor(input_size, cfg=cfg, is_multi=False)

            self.mlp_adapter = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2, bias=True),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size, bias=True),
                nn.ReLU(),
            )

            # for p in self.attention.parameters():
            #     p.requires_grad = False

        # self.attention = LV_selfttention(input_size, cfg=cfg, is_multi=False)
        # self.atten_bb = LV_attentionv2(input_size, cfg=cfg, is_multi=False)
        pass

    def forward_with_text_attention(self, fg_features, num_preds_per_image=None, gt_classes=None):
        # with torch.no_grad():
        #     roi_gt, roi_cls = self.get_roi_with_gt2(
        #         [features[f] for f in self.in_features], targets)
        # ot_feat_input = torch.cat(roi_gt, dim=0).unsqueeze(0)
        # ot_cls_input = torch.cat(roi_cls, dim=0).unsqueeze(0)

        feat = self.attention(
            fg_features, gt_classes, num_preds_per_image)
        # att = torch.cat(sim2stext, gim2gtext)
        # att = self.pro_att(att).softmax(dim=-1)
        # fg_features = self.atten_bb(fg_features, None)[0]
        # print('fg_features: ', fg_features.shape)
        # assert 0
        # att_ft = sim2stext+gim2gtext
        return feat

    def forward_adapter(self, fg_features, teacher_features):

        # feat = self.attention.forward_wo_label(fg_features)
        feat = self.mlp_adapter(fg_features)
        loss = {}
        alpha = 1.0
        margin = 0.2
        def norm_x(x): return F.normalize(x)

        def loss_cosine(a, b):
            return 1 - torch.einsum(
                'b i, b i -> b', norm_x(a), norm_x(b))

        if self.training and self.distill_mode:
            # l = ((feat - teacher_features)**2).mean()*0

            # l = loss_cosine(feat, teacher_features).mean()*alpha
            # l = (l - margin).clamp(min=0.2)
            # l[torch.where(l < margin)] = 0
            # l = l.mean()*alpha
            l = F.mse_loss(feat, teacher_features)*alpha

            # l = l/feat.shape[0]  # mean according batch size
            loss = {'loss_student_feat': l}

        return feat, loss

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

        # print('self.batch_size_per_image:', self.batch_size_per_image)
        # print('self.positive_sample_fraction:', self.positive_sample_fraction)
        # print('self.num_classes:', self.num_classes)
        from detectron2.layers import nonzero_tuple
        positive = nonzero_tuple(
            (gt_classes != -1) & (gt_classes != self.num_classes))[0]

        negative = nonzero_tuple(gt_classes == self.num_classes)[0]

        # sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
        #     gt_classes,
        #     self.batch_size_per_image,
        #     self.positive_sample_fraction,
        #     self.num_classes,
        # )

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
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._get_gt_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            # proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def forward_teacher(self, feature_pooled, proposals, test_with_gt=True):

        gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        num_preds_per_image = [len(p) for p in proposals]
        loss_att, output_att = self.forward_with_text_attention(
            feature_pooled, gt_classes=gt_classes, num_preds_per_image=num_preds_per_image)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled, output_att['sim2stext'])
        output_att['pred_logits'] = pred_class_logits
        output_att['pred_bbox'] = pred_proposal_deltas
        return output_att, loss_att

    def forward_student(self, feature_pooled, proposals, teacher_output):

        teacher_features = teacher_output.get('sim2stext', None)

        att_feature, loss = self.forward_adapter(
            feature_pooled, teacher_features=teacher_features)

        pred_class_logits, pred_proposal_deltas = self.stu_box_predictor(
            feature_pooled, att_feature)

        if self.student_training and self.training and self.distill_mode:
            t_logits = teacher_output['pred_logits']
            # if self.novel_tuning:
            #     t_logits, _ = self.stu_box_predictor(
            #         feature_pooled, teacher_features)

            params = {
                'alpha': 1,
                'temperature': 1,
            }
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)

            # compute loss value
            loss_kl = loss_fn_kd_only(outputs=pred_class_logits,
                                      labels=gt_classes,
                                      bg_label=self.num_classes,
                                      teacher_outputs=t_logits,
                                      params=params)
            loss.update({'loss_kl': loss_kl})

        output = {
            'pred_logits': pred_class_logits,
            'pred_bbox': pred_proposal_deltas
        }
        return output, loss

    def forward(self, images, features, proposals, targets=None):
        del images
        test_with_gt = True if (not self.training) and targets else False
        # print('test_with_gt:', test_with_gt)
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        elif test_with_gt:  # only use for teacher
            proposals = self.label_proposals(proposals, targets)
            # gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        t_output = {}
        if self.teacher_training:
            t_output, t_loss = self.forward_teacher(
                feature_pooled, proposals, test_with_gt)
            t_loss = {key+'_t': val for key, val in t_loss.items()}

            # t_loss = {key+'_t' : val for key, val in t_loss.items()}
            teacher_outputs = FastRCNNOutputs(
                self.box2box_transform,
                t_output['pred_logits'],
                t_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.student_training:
            s_output, s_loss = self.forward_student(
                feature_pooled, proposals, t_output)

            student_outputs = FastRCNNOutputs(
                self.box2box_transform,
                s_output['pred_logits'],
                s_output['pred_bbox'],
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            losses = {}
            if self.teacher_training:
                loss = teacher_outputs.losses()
                loss = {key+'_t': val for key, val in loss.items()}

                losses.update(loss)
                losses.update(t_loss)
            if self.student_training:
                losses.update(student_outputs.losses())
                losses.update(s_loss)

            return [], losses
        else:
            if self.teacher_training:
                pred_instances, _ = teacher_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            if self.student_training:
                pred_instances, _ = student_outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
            return pred_instances, {}
