EXP_NAME="distillate_rpn_l1_norm"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=1

# res4 only
cfg_MODEL="
   MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
   MODEL.ADDITION.NAME glove
   SOLVER.IMS_PER_BATCH 12
"
# multi-scale resnets features
# cfg_MODEL="
#    MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
#    MODEL.BACKBONE.FREEZE_AT 1
#    MODEL.RESNETS.OUT_FEATURES ['res2','res3','res4']
#    MODEL.ADDITION.NAME glove
#    SOLVER.IMS_PER_BATCH 8
#    SOLVER.MAX_ITER 30000
# "


python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
