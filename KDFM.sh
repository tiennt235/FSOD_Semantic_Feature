EXP_NAME="KDFM"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=2,3


cfg_MODEL="
   MODEL.META_ARCHITECTURE KDRCNN
   MODEL.ROI_HEADS.NAME KDFMRes5ROIHeads
   MODEL.ADDITION.STUDENT_TRAINING True
   MODEL.ADDITION.DISTILL_MODE True
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT False
   SOLVER.IMS_PER_BATCH 8
"

python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
exit