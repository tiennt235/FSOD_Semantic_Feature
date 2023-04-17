# run file: bash bash/danhnt.sh ablations 1
EXP_NAME=$1
SPLIT_ID=$2

N_GPUS=1

IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth

SAVE_DIR=checkpoints/voc/${EXP_NAME}

cfg_MODEL="
MODEL.ROI_HEADS.NAME AuxRes5ROIHeads
MODEL.ROI_HEADS.OUTPUT_LAYER AuxFastRCNNOutputLayers
AUX_MODEL.SEMANTIC_DIM 300
AUX_MODEL.INFERENCE_WITH_AUX False
SOLVER.IMS_PER_BATCH 8
"

# python3 main.py --num-gpus ${N_GPUS}  --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
#    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                 \
#        OUTPUT_DIR ${SAVE_DIR} TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}


# ----------------------------- Model Preparation --------------------------------- #
# python3 tools/model_surgery.py --dataset voc --method randinit                                \
#     --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
#     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}

# ------------------------------ Novel Fine-tuning ------------------------------- #
BASE_WEIGHT=checkpoints/voc/${EXP_NAME}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do
  for shot in 1 2 3 5 10
  do
    cfg_MODEL="
    MODEL.ROI_HEADS.NAME AuxRes5ROIHeads
    MODEL.ROI_HEADS.OUTPUT_LAYER AuxFastRCNNOutputLayers
    AUX_MODEL.SEMANTIC_DIM 300
    AUX_MODEL.INFERENCE_WITH_AUX False
    SOLVER.IMS_PER_BATCH 8
    "
    python3 tools/create_config.py --dataset voc --config_root configs/voc/${EXP_NAME}               \
        --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
    CONFIG_PATH=configs/voc/${EXP_NAME}/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
    OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
    python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
        --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
               TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} ${cfg_MODEL}
    rm ${CONFIG_PATH}
  done
done

python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # surmarize all results