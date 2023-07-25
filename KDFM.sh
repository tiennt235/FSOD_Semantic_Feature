<<<<<<< HEAD
EXP_NAME="KDRPN_BinaryCE"
=======
EXP_NAME="KDFM_dev"
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
SPLIT_ID=1
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-101.pkl
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
<<<<<<< HEAD
# IMAGENET_PRETRAIN=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth # Backbone pretrained
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

=======
NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=2,3
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31

   # MODEL.PROPOSAL_GENERATOR.NAME KDRPN
cfg_MODEL="
<<<<<<< HEAD
    MODEL.META_ARCHITECTURE KDRCNN
    MODEL.AUX.NAME glove
    MODEL.PROPOSAL_GENERATOR.NAME KDRPN
=======
   MODEL.META_ARCHITECTURE KDRCNN
   MODEL.ROI_HEADS.NAME KDFMRes5ROIHeads
   MODEL.ADDITION.TEACHER_TRAINING False
   MODEL.ADDITION.STUDENT_TRAINING True
   MODEL.ADDITION.DISTILL_ON True
   MODEL.ADDITION.NAME glove
   MODEL.ADDITION.INFERENCE_WITH_GT False
   MODEL.ADDITION.KD_TEMP 5
   SOLVER.IMS_PER_BATCH 8
   SOLVER.MAX_ITER 30000
   SOLVER.CHECKPOINT_PERIOD 15000
>>>>>>> f39460a156536f65f659a3ff33ff8db22da8ad31
"
# python3 tools/model_surgery.py --dataset voc --method remove                                                                    \
#     --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
#     --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
# exit

# cfg_MODEL="
#    MODEL.META_ARCHITECTURE KDRCNN
#    MODEL.ROI_HEADS.NAME KDFMRes5ROIHeads
#    MODEL.PROPOSAL_GENERATOR.NAME KDRPN
#    MODEL.ADDITION.TEACHER_TRAINING False
#    MODEL.ADDITION.STUDENT_TRAINING True
#    MODEL.ADDITION.DISTILL_ON False
#    MODEL.ADDITION.NAME glove
#    MODEL.ADDITION.INFERENCE_WITH_GT False
#    MODEL.ADDITION.KD_TEMP 5
#    SOLVER.IMS_PER_BATCH 8
#    SOLVER.MAX_ITER 30000
#    SOLVER.CHECKPOINT_PERIOD 15000
# "

python3 main.py --num-gpus ${NUM_GPUS} --dist-url auto --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml \
   --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN} \
   OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID} ${cfg_MODEL}
exit

# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                                                    \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 #2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${NUM_GPUS} --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} ${cfg_MODEL}
                   
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done