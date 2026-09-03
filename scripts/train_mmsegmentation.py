"""
Train a mmseg-based segmentation model (generally used for ViT backbones, e.g. Mask2Former with DINOv2 backbone).

Requires --pth_backbone_checkpoint to load an existing backbone checkpoint for transfer learning (default: None, so no checkpoint is loaded).
Choice between binary (default) or multi-class modes.
Choice between various other parameters (see argparse help for details).

Model paths are saved to data/models/<model_number>_seg.pth, and results are saved to data/results/segmentation.csv.
Logging is saved to data/logs/<model_number>/.

To execute this with distributed training, run:
    PYTHONUNBUFFERED=1 nohup .venv/bin/torchrun --standalone --nproc_per_node=2 \
        python scripts/train_mmsegmentation.py --str_architecture <architecture> --str_encoder <encoder> \
        [--pth_backbone_checkpoint <path_to_checkpoint>] [--int_size <image_size>] [--flt_lr <learning_rate>] [--int_epochs <num_epochs>] \
        [--int_batch <batch_size>] [--int_workers <num_workers>] [--bl_no_shuffle] [--bl_no_weighted_sampler]
For example:
    --str_architecture mask2former --str_encoder dinov2 --pth_backbone_checkpoint models/1_clf.ckpt --int_size 518 --flt_lr 1e-4 \
        --int_epochs 25 --int_batch 16 --int_workers 4 --bl_no_shuffle --bl_no_weighted_sampler
"""
import argparse
import os
import shutil
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics import F1Score

import utils
from dataloaders import initialise_dataloaders
from models_classification import ClfInitialisers as ClfInitialisers
from models_segmentation import SegInitialisers as SegInitialisers, initialise_mmseg

# mmsegmentation imports must be done after the above imports to avoid mmcv version conflicts
import mmcv
_str_mmcv_version: str = mmcv.__version__
mmcv.__version__ = "2.1.0" # installed mmcv (2.2.0) is API-compatible but newer than mmseg/mmdet"s asserted upper bound
import mmseg.models # noqa: F401 - registers all backbones/necks/decode_heads (incl. mmdet-backed Mask2FormerHead)
from mmengine.config import ConfigDict
from mmengine.dist import get_rank, init_dist
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.hooks import Hook
from mmengine.registry import HOOKS, METRICS
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
mmcv.__version__ = _str_mmcv_version

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a mmseg-based segmentation model.")
    parser.add_argument("--str_architecture", type=str, default="mask2former", choices=utils.Models.lst_mmseg_architectures,
                        help="Architecture to use for the segmentation model (default: mask2former).")
    parser.add_argument("--str_encoder", type=str, default="dinov2", choices=utils.Models.lst_mmseg_encoders,
                        help="Encoder backbone to use for the encoder-decoder architecture (default: dinov2)")
    parser.add_argument("--pth_backbone_checkpoint", type=str, default=None,
                        help="Path to a checkpoint for the backbone encoder (default: None, so no checkpoint is loaded)")
    parser.add_argument("--int_size", type=int, default=518, choices=[518],
                        help="Spatial size for images and masks (default: 518 for DINOv2 backbones)")

    parser.add_argument("--flt_lr", type=float, default=1e-4,
                        help="Learning rate for training models (default: 1e-4)")

    parser.add_argument("--int_epochs", type=int, default=25,
                        help="Number of training epochs per phase (default: 25)")
    parser.add_argument("--int_batch", type=int, default=16,
                        help="Batch size for training per phase (default: 16)")
    parser.add_argument("--int_workers", type=int, default=4,
                        help="Number of workers for classifier data loading (default: 4)")

    parser.add_argument("--bl_no_shuffle", action="store_true",
                        help="Whether to shuffle the training dataloader (default: False, so shuffling is enabled).")
    parser.add_argument("--bl_no_weighted_sampler", action="store_true",
                        help="Whether to use the weighted sampler for the training dataloader (default: False, so weighted sampler is enabled).")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main training entry point for segmentation."""
    logger.info("=" * 70)
    logger.info("LOADING TRAINING CONFIGURATION...")
    args: argparse.Namespace = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    str_launcher: str = "pytorch" if "LOCAL_RANK" in os.environ else "none"
    if str_launcher != "none":
        init_dist(str_launcher) # initializes distributed training (torchrun/torch.distributed.launch sets LOCAL_RANK)
    register_all_modules(init_default_scope=True) # registers all modules in mmseg, including backbones, necks, decode_heads, and datasets
    if get_rank() == 0:
        int_model: int = utils.record_model_parameters(dct_parameters=vars(args), bl_record=True)
    else:
        time.sleep(5) # allow rank 0 time to write to ledger
        int_model: int = utils.record_model_parameters(dct_parameters=vars(args), bl_record=False)

    logger.info("=" * 70)
    logger.info("LOADING SUPERVISED DATA...")
    int_classes: int = utils.Instruments.int_classes
    dct_dataloaders: dict[str, DataLoader] = initialise_dataloaders(
        str_task="seg",
        bl_multiclass=True, # binary not supported
        int_batch=args.int_batch,
        int_size=args.int_size,
        int_workers=args.int_workers,
        bl_shuffle=not args.bl_no_shuffle,
        bl_weighted_sampler=not args.bl_no_weighted_sampler,
        bl_mmseg=True,
    )

    logger.info("=" * 70)
    logger.info(f"LOADING SUPERVISED MODEL...")
    model = ConfigDict(
        type="EncoderDecoder",
        data_preprocessor=dict(type="SegDataPreProcessor", mean=None, std=None, bgr_to_rgb=False, pad_val=0, seg_pad_val=255, size=(args.int_size, args.int_size)), # images already normalized to [0, 1] in dataloader
        backbone=SegInitialisers.backbone_mmseg(str_backbone=args.str_encoder, str_backbone_checkpoint=args.pth_backbone_checkpoint),
        neck=SegInitialisers.neck_mmseg(str_architecture=args.str_architecture),
        decode_head=initialise_mmseg(str_architecture=args.str_architecture, int_classes=int_classes),
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
    )
    optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(type="AdamW", lr=args.flt_lr, weight_decay=args.flt_lr * 100), 
        clip_grad=dict(max_norm=0.01, norm_type=2),
        paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}), # Head trains at 1e-4, Backbone trains at 1e-5 (1/10th)
    )
    param_scheduler = [
        dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1000),
        dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1000, end=40000, by_epoch=False),
    ]

    runner = Runner(
        model=model,
        default_scope="mmseg",
        launcher=str_launcher,
        work_dir=str(utils.Directories.pth_logs / str(int_model)),
        train_dataloader=dct_dataloaders["train"],  
        val_dataloader=dct_dataloaders["val"],
        test_dataloader=dct_dataloaders["test"],
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=25, val_interval=1),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        val_evaluator=[
            dict(type="IoUMetric", iou_metrics=["mDice"]),
            dict(type="FrameF1Metric", num_classes=int_classes),
        ],
        test_evaluator=[
            dict(type="IoUMetric", iou_metrics=["mDice"]),
            dict(type="FrameF1Metric", num_classes=int_classes),
        ],
        
        visualizer=dict(
            type="SegLocalVisualizer",
            vis_backends=[
                dict(type="LocalVisBackend"), # keeps a backup of text logs in work_dirs
                dict(type="TensorboardVisBackend"),
            ],
        ),
        
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=1, max_keep_ckpts=1, save_best="mDice", rule="greater", out_dir=str(utils.Directories.pth_models)),
            train_metric=dict(type="TrainMetricHook", evaluator=[
                dict(type="IoUMetric", iou_metrics=["mDice"]),
                dict(type="FrameF1Metric", num_classes=int_classes),
            ]),
        )
    )

    logger.info("=" * 70)
    logger.info(f"TRAINING SUPERVISED MODEL...")
    runner.train()

    logger.info("=" * 70)
    logger.info("EVALUATING SUPERVISED RESULTS...")
    pth_best_ckpt: Path = utils.Directories.pth_models / f"{int_model}_seg.pth"

    if get_rank() == 0:
        str_best_ckpt_old: str = runner.message_hub.get_info("best_ckpt")
        if get_rank() == 0:
            shutil.copy(str_best_ckpt_old, pth_best_ckpt)
            evaluate_best_checkpoint(
                runner=runner,
                dct_dataloaders=dct_dataloaders,
                int_classes=int_classes,
                int_model=int_model,
                str_model=args.str_architecture,
                pth_best_ckpt=pth_best_ckpt,
            )

    logger.info("TRAINING COMPLETE!!!")
    logger.info("=" * 70)


def evaluate_best_checkpoint(runner: Runner, dct_dataloaders: dict[str, DataLoader], int_classes: int, int_model: int, str_model: str, pth_best_ckpt: Path) -> None:
    """
    Loads the best checkpoint (tracked by CheckpointHook's save_best) and evaluates it on the train,
    val, and test splits, reusing the same metric definitions as models_segmentation.py's Initialisers
    (dice, dice_binary) and models_classification.py's Initialisers (accuracy, balanced_accuracy,
    precision, recall, f1_macro, f1, f1_binary) - the latter applied to the frame-level dominant class
    (mode of non-zero pixels) extracted from each predicted/ground-truth segmentation mask.
    Args:
        runner (Runner): The mmengine Runner object used for training.
        dct_dataloaders (dict[str, DataLoader]): Dictionary containing the dataloaders for train, val, and test splits.
        int_classes (int): Number of classes in the segmentation task.
        int_model (int): Model number for logging and saving results.
        str_model (str): Model architecture name for logging and saving results.
        pth_best_ckpt (Path): Path to the best checkpoint file to be evaluated.
    Returns:
        None but saves the evaluation results to a CSV file and logs the metrics.
    """
    runner.load_checkpoint(str(pth_best_ckpt))
    runner.model.eval()

    for str_split in ["train", "val", "test"]:
        metrics_seg = SegInitialisers.metrics(int_classes=int_classes)[str_split]
        metrics_seg_binary = SegInitialisers.metrics_binary()[str_split]
        metrics_clf = ClfInitialisers.metrics(int_classes=int_classes)[str_split]
        metrics_clf_binary = ClfInitialisers.metrics_binary()[str_split]

        for metrics_collection in (metrics_seg, metrics_seg_binary, metrics_clf, metrics_clf_binary):
            for metric in metrics_collection.values():
                metric.sync_on_compute = False # prevents running on other ranks

        lst_pred_classes: list[int] = []
        lst_gt_classes: list[int] = []
        with torch.no_grad():
            for data_batch in dct_dataloaders[str_split]:
                for data_sample in runner.model.val_step(data_batch):
                    tsr_pred: torch.Tensor = data_sample.pred_sem_seg.data.squeeze(0)
                    tsr_gt: torch.Tensor = data_sample.gt_sem_seg.data.squeeze(0).long()

                    metrics_seg.update(tsr_pred.unsqueeze(0), tsr_gt.unsqueeze(0))
                    metrics_seg_binary.update((tsr_pred > 0).long().unsqueeze(0).unsqueeze(0), (tsr_gt > 0).long().unsqueeze(0).unsqueeze(0))

                    lst_pred_classes.append(int(FrameF1Metric.dominant_class(tsr_pred).item()))
                    lst_gt_classes.append(int(FrameF1Metric.dominant_class(tsr_gt).item()))

        tsr_preds_clf = torch.clamp(torch.tensor(lst_pred_classes), 0, int_classes - 1)
        tsr_gt_clf = torch.clamp(torch.tensor(lst_gt_classes), 0, int_classes - 1)
        metrics_clf.update(tsr_preds_clf, tsr_gt_clf)
        metrics_clf_binary.update((tsr_preds_clf > 0).long(), (tsr_gt_clf > 0).long())

        dct_results: dict = {
            **{str_key: tsr_value.item() for str_key, tsr_value in metrics_seg.compute().items()},
            **{str_key: tsr_value.item() for str_key, tsr_value in metrics_seg_binary.compute().items()},
            **{str_key: tsr_value.item() for str_key, tsr_value in metrics_clf.compute().items()},
            **{str_key: tsr_value.item() for str_key, tsr_value in metrics_clf_binary.compute().items()},
        }
        utils.save_results(
            results=[dct_results],
            int_model=int_model,
            str_model=str_model,
            str_class="multi",
            str_split=str_split,
            pth_results=utils.Directories.pth_results_seg,
        )
        logger.info(f"Results for {str_split} split:")
        for str_key, value in dct_results.items():
            logger.info(f"{str_key}: {value}")


@METRICS.register_module()
class FrameF1Metric(BaseMetric):
    """
    Frame-level macro F1: reduces each frame"s per-pixel prediction/ground-truth to a single dominant (mode) non-zero class, 
    then computes multiclass macro F1 across frames (background class 0 excluded). 
    Mirrors the frame-level classification metric in models_segmentation.py,
    as opposed to mmseg"s IoUMetric mFscore which is a per-pixel/per-class metric.
    """

    def __init__(self, num_classes: int, collect_device: str = "cpu", prefix: str | None = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes

    @staticmethod
    def dominant_class(tsr_class_map: torch.Tensor) -> torch.Tensor:
        tsr_flat = tsr_class_map.flatten()
        tsr_nonzero = tsr_flat[tsr_flat != 0]
        if tsr_nonzero.numel() > 0:
            return tsr_nonzero.mode().values
        return torch.tensor(0, device=tsr_flat.device, dtype=torch.long)

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            int_pred_class = self.dominant_class(data_sample["pred_sem_seg"]["data"]).item()
            int_gt_class = self.dominant_class(data_sample["gt_sem_seg"]["data"]).item()
            self.results.append((int_pred_class, int_gt_class))

    def compute_metrics(self, results: list) -> dict:
        tsr_preds = torch.tensor([int_pred for int_pred, _ in results])
        tsr_targets = torch.tensor([int_gt for _, int_gt in results])
        f1_score = F1Score(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=0)
        return {"frameF1": round(f1_score(tsr_preds, tsr_targets).item() * 100, 2)}


@HOOKS.register_module()
class TrainMetricHook(Hook):
    """
    Runs evaluator over the training set at the end of every epoch and prints the result
    (mmseg"s IoUMetric only runs on val/test loops otherwise).
    """

    def __init__(self, evaluator: dict):
        self.evaluator_cfg = evaluator

    def after_train_epoch(self, runner) -> None:
        evaluator = Evaluator(self.evaluator_cfg)
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, "metainfo"):
            evaluator.dataset_meta = dataset.metainfo
        runner.model.eval()
        with torch.no_grad():
            for data_batch in runner.train_dataloader:
                outputs = runner.model.val_step(data_batch)
                evaluator.process(data_samples=outputs, data_batch=data_batch)
        metrics = evaluator.evaluate(len(dataset))
        runner.model.train()
        runner.logger.info(f"Epoch(train) [{runner.epoch}] {metrics}")


if __name__ == "__main__":
    main()
