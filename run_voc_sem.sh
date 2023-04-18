# EXP_NAME="semantic_rpn_roi_heads_x1.2_bbox"
EXP_NAME="dev"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
IMAGENET_PRETRAIN=checkpoints/voc/semantic_rpn_roi_heads_x1.2_bbox/defrcn_det_r101_base1/model_final.pth
NUM_GPUS=1


# semantic_rpn
# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedTextRCNN
# SOLVER.IMS_PER_BATCH 8
# "

# semantic_rpn_roi_heads
cfg_MODEL="
MODEL.META_ARCHITECTURE GeneralizedSemanticRCNN
MODEL.RPN.ADDITION True
MODEL.ROI_HEADS.NAME SemanticRes5ROIHeads
MODEL.DISTILLATION.TEACHER_TRAINING True
MODEL.DISTILLATION.STUDENT_TRAINING False
SOLVER.IMS_PER_BATCH 8
"

python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}

exit