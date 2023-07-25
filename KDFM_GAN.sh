EXP_NAME="KDFM_dev"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
IMAGENET_PRETRAIN=checkpoints/voc/concat_sem_x1.2_rpn/defrcn_det_r101_base1/model_final.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


cfg_MODEL="
   MODEL.META_ARCHITECTURE KDRCNN
   MODEL.ROI_HEADS.NAME KDFMRes5ROIHeads
   MODEL.AUX.TEACHER_TRAINING False
   MODEL.AUX.STUDENT_TRAINING True
   MODEL.AUX.DISTILL_ON True
   MODEL.AUX.DISTILL_FEATURES ['res4']
   MODEL.AUX.NAME glove
   MODEL.AUX.INFERENCE_WITH_GT False
   MODEL.AUX.KD_TEMP 5
   SOLVER.IMS_PER_BATCH 12
"

   # --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth \
python3 tools/model_surgery.py --dataset voc --method reset \
   --src-path ${IMAGENET_PRETRAIN} \
   --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
TEACHER_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset.pth

python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${TEACHER_WEIGHT} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
exit

python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}