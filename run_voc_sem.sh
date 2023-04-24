# EXP_NAME="semantic_rpn_roi_heads_x1.2_bbox"
EXP_NAME="attention_dev"
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
NUM_GPUS=1


# semantic_rpn
# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedTextRCNN
# SOLVER.IMS_PER_BATCH 8
# "

# semantic_rpn_roi_heads
# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedSemanticRCNN
# MODEL.RPN.ADDITION True
# MODEL.ROI_HEADS.NAME SemanticRes5ROIHeads
# MODEL.DISTILLATION.TEACHER_TRAINING True
# MODEL.DISTILLATION.STUDENT_TRAINING False
# SOLVER.IMS_PER_BATCH 8
# "

# attention_dev
# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedSemanticRCNN
# MODEL.ROI_HEADS.NAME SemanticRes5ROIHeads
# MODEL.DISTILLATION.TEACHER_TRAINING True
# MODEL.DISTILLATION.STUDENT_TRAINING True
# MODEL.DISTILLATION.MODE True
# MODEL.DISTILLATION.L2 True
# SOLVER.IMS_PER_BATCH 12
# "
# cfg_MODEL="
# MODEL.ROI_HEADS.NAME SemanticRes5ROIHeads
# SOLVER.IMS_PER_BATCH 12
# "
# python3 main.py --num-gpus ${NUM_GPUS} --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
#    OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
# exit
# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset voc --method remove                                    \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth

# for seed in 0 1 2 3 4 5 6 7 8 9
for seed in 2 3 4 5 6 7 8 9
do
    for shot in 1 #2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
      cfg_MODEL="
      MODEL.ROI_HEADS.NAME SemanticRes5ROIHeads
      SOLVER.IMS_PER_BATCH 12
      "
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
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/ --shot-list 1 #2 3 5 10  # surmarize all results