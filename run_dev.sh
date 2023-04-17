# run file: bash bash/danhnt.sh ablations 1
EXP_NAME=$1
SPLIT_ID=$2

N_GPUS=1

IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=/home/hoangpn/Ecai/DeFRCN_tien/checkpoints/voc/AAAI/teacher_base/defrcn_det_r101_base1/model_final.pth

SAVE_DIR=checkpoints/voc/${EXP_NAME}
TEACHER_PATH=checkpoints/voc/${EXP_NAME}/teacher_base/defrcn_det_r101_base${SPLIT_ID}

# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedRCNN1
# MODEL.ROI_HEADS.NAME TextRes5ROIHeads
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING False
# MODEL.ROI_HEADS.DISTILLATE True
# SOLVER.IMS_PER_BATCH 8
# SOLVER.MAX_ITER 10000
# SOLVER.CHECKPOINT_PERIOD 5000
# TEST.EVAL_PERIOD 5000
# "

# cfg_MODEL="
# MODEL.META_ARCHITECTURE GeneralizedRCNN1
# MODEL.ROI_HEADS.NAME TextRes5ROIHeads
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING False
# SOLVER.IMS_PER_BATCH 4
# "

cfg_MODEL="
MODEL.ROI_HEADS.NAME AuxRes5ROIHeads
MODEL.ROI_HEADS.OUTPUT_LAYER AuxFastRCNNOutputLayers
AUX_MODEL.SEMANTIC_DIM 300
AUX_MODEL.INFERENCE_WITH_AUX False
SOLVER.IMS_PER_BATCH 8
"

python3 main.py --num-gpus ${N_GPUS}  --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
       OUTPUT_DIR ${TEACHER_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

exit
# python3 tools/model_surgery.py --dataset voc --method reset                                \
#    --src-path ${TEACHER_PATH}/model_final.pth                    \
#    --save-dir ${TEACHER_PATH}

# python3 tools/model_surgery.py --dataset voc --method randinit                                \
#    --src-path ${TEACHER_PATH}/model_final.pth                    \
#    --save-dir ${TEACHER_PATH}
# exit
# SOLVER:
#   IMS_PER_BATCH: 16
#   BASE_LR: 0.01
#   STEPS: (640, )
#   MAX_ITER: 800
#   CHECKPOINT_PERIOD: 100000
#   WARMUP_ITERS: 0

# BASE_WEIGHT=${TEACHER_PATH}/model_reset_surgery.pth
# for shot in 1 # 1 2 3 5 10  # if final, 10 -> 1 2 3 5 10
# do
#     for seed in  10 1 2 #3 4 5 6 7 8 9 # 10
#     do
#     cfg_MODEL="
#     MUTE_HEADER True
#     MODEL.ROI_HEADS.NAME TextRes5ROIHeads
#     MODEL.ROI_HEADS.TEACHER_TRAINING True
#     MODEL.ROI_HEADS.STUDENT_TRAINING False
#     MODEL.ROI_HEADS.DISTILLATE False
#     "
#     # TEST.EVAL_PERIOD 5000
#     python3 tools/create_config.py --dataset voc --config_root configs/voc               \
#         --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
#     CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
#     CURRENT_WEIGHT=${BASE_WEIGHT}

#     OUTPUT_DIR=${SAVE_DIR}/teacher_novel3/${shot}shot_seed${seed}


#     # python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file ${CONFIG_PATH}                            \
#     #     --opts MODEL.WEIGHTS ${CURRENT_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
#     #         TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

#     python3 tools/model_surgery.py --dataset voc --method reset                                \
#    --src-path ${OUTPUT_DIR}/model_final.pth                    \
#    --save-dir ${OUTPUT_DIR}

#     cfg_MODEL="
#     MUTE_HEADER True
#     MODEL.ROI_HEADS.NAME TextRes5ROIHeads
#     MODEL.ROI_HEADS.TEACHER_TRAINING False
#     MODEL.ROI_HEADS.STUDENT_TRAINING True
#     MODEL.ROI_HEADS.DISTILLATE True
#     MODEL.ROI_HEADS.L2 False
#     MODEL.ROI_HEADS.KL_TEMP 5
#     SOLVER.MAX_ITER 3000
#     "
#     # TEST.EVAL_PERIOD 5000 checkpoints/voc/text+visual10_old/teacher_novel/1shot_seed1/model_reset_optimizer.pth
#     CURRENT_WEIGHT=${OUTPUT_DIR}/model_reset_optimizer.pth
#     OUTPUT_DIR=${SAVE_DIR}/student_novel3_x1.5iter/${shot}shot_seed${seed}

#     python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file ${CONFIG_PATH}                            \
#         --opts MODEL.WEIGHTS ${CURRENT_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
#             TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL} 

#     rm ${CONFIG_PATH}
#     done

# done
# exit
# for shot in 2 # 3 5 10  # if final, 10 -> 1 2 3 5 10
# do
#     for seed in  10 2 3 4 5 6 7 8 9 # 10
#     do
#     cfg_MODEL="
#     MUTE_HEADER True
#     MODEL.META_ARCHITECTURE GeneralizedRCNN2
#     MODEL.ROI_HEADS.NAME SuperRes5ROIHeads2
#     MODEL.ROI_HEADS.TEACHER_TRAINING True
#     MODEL.ROI_HEADS.STUDENT_TRAINING False
#     MODEL.ROI_HEADS.DISTILLATE False
#     "
#     # TEST.EVAL_PERIOD 5000
#     python3 tools/create_config.py --dataset voc --config_root configs/voc               \
#         --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
#     CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
#     CURRENT_WEIGHT=${BASE_WEIGHT}



#     OUTPUT_DIR=${SAVE_DIR}/teacher_novel/${shot}shot_seed${seed}


#     python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file ${CONFIG_PATH}                            \
#         --opts MODEL.WEIGHTS ${CURRENT_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
#             TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}

#     python3 tools/model_surgery.py --dataset voc --method reset                                \
#    --src-path ${OUTPUT_DIR}/model_final.pth                    \
#    --save-dir ${OUTPUT_DIR}

#     cfg_MODEL="
#     MUTE_HEADER True
#     MODEL.META_ARCHITECTURE GeneralizedRCNN2
#     MODEL.ROI_HEADS.NAME SuperRes5ROIHeads2
#     MODEL.ROI_HEADS.TEACHER_TRAINING False
#     MODEL.ROI_HEADS.STUDENT_TRAINING True
#     MODEL.ROI_HEADS.DISTILLATE True
#     "
#     # TEST.EVAL_PERIOD 5000 checkpoints/voc/text+visual10_old/teacher_novel/1shot_seed1/model_reset_optimizer.pth
#     CURRENT_WEIGHT=${OUTPUT_DIR}/model_reset_optimizer.pth
#     OUTPUT_DIR=${SAVE_DIR}/student_novel/${shot}shot_seed${seed}

#     python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file ${CONFIG_PATH}                            \
#         --opts MODEL.WEIGHTS ${CURRENT_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
#             TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL} 

#     rm ${CONFIG_PATH}
#     done

# done

# exit

# cfg_MODEL="
# MUTE_HEADER True
# MODEL.META_ARCHITECTURE GeneralizedRCNN2
# MODEL.ROI_HEADS.NAME TextRes5ROIHeads2
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING False
# MODEL.ROI_HEADS.DISTILLATE False
# TEST.EVAL_PERIOD 5000
# "
# # SOLVER.BASE_LR 0.02

# python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
#        OUTPUT_DIR ${TEACHER_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}


# python3 tools/model_surgery.py --dataset voc --method reset                                \
#    --src-path ${TEACHER_PATH}/model_final.pth                    \
#    --save-dir ${TEACHER_PATH}
# exit
# # train student model
# cfg_MODEL='
# MUTE_HEADER True
# MODEL.META_ARCHITECTURE GeneralizedRCNN2
# MODEL.ROI_HEADS.NAME TextRes5ROIHeads2
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING True
# MODEL.ROI_HEADS.DISTILLATE True
# TEST.EVAL_PERIOD 500000
# MODEL.ROI_HEADS.L2 False
# MODEL.ROI_HEADS.KL True
# '

# STUDENT_PATH=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
# python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${TEACHER_WEIGHT}                                 \
#        OUTPUT_DIR ${STUDENT_PATH} ${cfg_MODEL}





# ####################################################################################################################################################################################

# EXP_NAME=text+visual4
# SPLIT_ID=1

# # N_GPUS=4
# # export CUDA_VISIBLE_DEVICES=4,5,6,7

# SAVE_DIR=checkpoints/voc/${EXP_NAME}
# IMAGENET_PRETRAIN=weights/R-101.pkl
# IMAGENET_PRETRAIN_TORCH=weights/resnet101-5d3b4d8f.pth 


# # train teacher model

# TEACHER_PATH=${SAVE_DIR}/teacher_base/defrcn_det_r101_base1

# cfg_MODEL="
# MUTE_HEADER True
# MODEL.META_ARCHITECTURE GeneralizedRCNN2
# MODEL.ROI_HEADS.NAME SuperRes5ROIHeads2
# MODEL.ROI_HEADS.TEACHER_TRAINING True
# MODEL.ROI_HEADS.STUDENT_TRAINING False
# MODEL.ROI_HEADS.DISTILLATE False
# "

# python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
#        OUTPUT_DIR ${TEACHER_PATH} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}


# python3 tools/model_surgery.py --dataset voc --method reset                                \
#    --src-path ${TEACHER_PATH}/model_final.pth                    \
#    --save-dir ${TEACHER_PATH}
TEACHER_WEIGHT=${TEACHER_PATH}/model_reset_surgery.pth
# exit
# train student model
cfg_MODEL='
MUTE_HEADER True
MODEL.META_ARCHITECTURE GeneralizedRCNN1
MODEL.ROI_HEADS.NAME TextRes5ROIHeads
MODEL.ROI_HEADS.TEACHER_TRAINING True
MODEL.ROI_HEADS.STUDENT_TRAINING True
MODEL.ROI_HEADS.DISTILLATE True
TEST.EVAL_PERIOD 500000
MODEL.ROI_HEADS.L2 False
MODEL.ROI_HEADS.KL True
SOLVER.IMS_PER_BATCH 8
'

STUDENT_PATH=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
python3 main.py --num-gpus ${N_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
   --opts MODEL.WEIGHTS ${TEACHER_WEIGHT}                                 \
       OUTPUT_DIR ${STUDENT_PATH} ${cfg_MODEL}


