EXP_NAME="concat_sem_x1.2_rpn_concat_roi_heads"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# res4 only
cfg_MODEL="
   MODEL.META_ARCHITECTURE SemanticRCNN
   MODEL.ROI_HEADS.NAME DistillatedRes5ROIHeads
   MODEL.ADDITION.TEACHER_TRAINING True
   MODEL.ADDITION.STUDENT_TRAINING False
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT True
"

# multi-scale resnets features
# cfg_MODEL="
#    MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
#    MODEL.BACKBONE.FREEZE_AT 3
#    MODEL.RESNETS.OUT_FEATURES ['res3','res4']
#    MODEL.ADDITION.NAME glove
# "


python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
