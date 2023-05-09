EXP_NAME="concat_sem_x1.2_rpn"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7


# multi-scale resnets features
# cfg_MODEL="
#    MODEL.META_ARCHITECTURE GeneralizedDistillatedRCNN
#    MODEL.BACKBONE.FREEZE_AT 3
#    MODEL.RESNETS.OUT_FEATURES ['res3','res4']
#    MODEL.ADDITION.NAME glove
# "

# ==================== teacher training ====================
cfg_MODEL="
   MODEL.META_ARCHITECTURE KDRCNN
   MODEL.ADDITION.TEACHER_TRAINING True
   MODEL.ADDITION.STUDENT_TRAINING False
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT True
"

python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
exit
# ==================== distillate to student ====================
python3 tools/model_surgery.py --dataset voc --method reset \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
TEACHER_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset.pth

cfg_MODEL="
   MODEL.META_ARCHITECTURE SemanticRCNN
   MODEL.ROI_HEADS.NAME DistillatedRes5ROIHeads
   MODEL.ROI_HEADS.FREEZE_BOX_PREDICTOR True
   MODEL.ADDITION.TEACHER_TRAINING False
   MODEL.ADDITION.STUDENT_TRAINING True
   MODEL.ADDITION.DISTILL_MODE True
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT False
"

STUDENT_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/student/model_final.pth
python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${TEACHER_WEIGHT} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/student_freeze_teacher ${cfg_MODEL}
exit

# ==================== G-FSOD Fine-tuning ====================

python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/student/model_final.pth                    \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/student
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/student/model_reset_remove.pth

cfg_MODEL="
   MODEL.META_ARCHITECTURE SemanticRCNN
   MODEL.ROI_HEADS.NAME DistillatedRes5ROIHeads
   MODEL.ADDITION.TEACHER_TRAINING False
   MODEL.ADDITION.STUDENT_TRAINING True
   MODEL.ADDITION.DISTILL_MODE False
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT False
   MODEL.ADDITION.KL_TEMP 1
"
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 #2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc                \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${NUM_GPUS} --config-file ${CONFIG_PATH}                             \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                      \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}
        rm ${CONFIG_PATH}
      #   rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/ --shot-list 1 
