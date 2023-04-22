EXP_NAME="distillate_rpn_preserve_bg_clip"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
NUM_GPUS=1


cfg_MODEL="
MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
MODEL.RPN.ADDITION_MODEL clip
SOLVER.IMS_PER_BATCH 12
"

python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
