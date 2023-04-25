EXP_NAME="distillate_rpn_res234"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


cfg_MODEL="
   MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
   MODEL.RESNETS.OUT_FEATURES ['res2','res3','res4'] 
   MODEL.ADDITION.NAME glove
"

python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
