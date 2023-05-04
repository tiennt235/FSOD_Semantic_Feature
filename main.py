import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from detectron2.config import CfgNode as CN
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup
import torch


def add_new_configs(cfg):
    cfg.MODEL.ADDITION = CN()
    cfg.MODEL.ADDITION.NAME = None
    cfg.MODEL.ADDITION.INFERENCE_WITH_GT = False
    cfg.MODEL.ADDITION.TEACHER_TRAINING = False
    cfg.MODEL.ADDITION.STUDENT_TRAINING = False
    cfg.MODEL.ADDITION.DISTIL_MODE = False
    
def batch_size_based_cfg_adjustment(cfg):
    alpha = 16 / cfg.SOLVER.IMS_PER_BATCH
    # alpha = 1
    cfg.defrost()
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/alpha
    cfg.SOLVER.STEPS = tuple([int(step*alpha) for step in cfg.SOLVER.STEPS])
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER*alpha)

def align_iter_student(cfg):
    alpha = 1
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR/alpha
    # alpha = 2
    cfg.SOLVER.STEPS = tuple([int(i/alpha) for i in cfg.SOLVER.STEPS])
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER/alpha)
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.WARMUP_ITERS/alpha)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(
                dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    add_new_configs(cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    ###
    batch_size_based_cfg_adjustment(cfg)
    ###
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
