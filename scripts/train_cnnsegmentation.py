"""
Train an encoder-decoder segmentation model.

Choice to load an existing backbone checkpoint for transfer learning (default: None, so no checkpoint is loaded).
Choice between binary (default) or multi-class modes.
Choice between distributed training (default) or single-GPU training (set --int_devices=1).
Choice between various other parameters (see argparse help for details).

Model paths are saved to data/models/<model_number>_seg.ckpt, and results are saved to data/results/segmentation.csv.
TensorBoard logs are saved to data/logs/<model_number>/.

To excecute this, run:
    python scripts/train_cnnsegmentation.py --str_architecture <architecture> --str_encoder <encoder> \
        [--pth_backbone_checkpoint <path_to_checkpoint>] [--bl_multiclass] [--int_size <image_size>] \
        [--int_epochs <num_epochs>] [--int_patience <early_stopping_patience>] [--int_batch <batch_size>] \
        [--int_epochs_freeze <num_epochs_freeze>] [--bl_no_shuffle] [--bl_no_weighted_sampler] [--flt_lr <learning_rate>] \
        [--int_workers <num_workers>] [--int_devices <num_gpus>]
For example:
    --str_architecture deeplabv3plus --str_encoder dinov2 --pth_backbone_checkpoint data/models/1_clf.ckpt \
        --bl_multiclass --int_size 518 --int_epochs 50 --int_patience 10 --int_batch 16 --int_epochs_freeze 10 \
        --bl_no_shuffle --bl_no_weighted_sampler --flt_lr 1e-3 --int_workers 4 --int_devices 1
"""
import argparse
import lightning as L
import time
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import utils
from dataloaders import initialise_dataloaders
from models_segmentation import CNNSegmenter

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")

    parser.add_argument("--str_architecture", type=str, default="deeplabv3plus", choices=utils.Models.lst_cnnseg_architectures,
                        help="Architecture to use for the segmentation model (default: deeplabv3plus).")
    parser.add_argument("--str_encoder", type=str, default="resnet50", choices=utils.Models.lst_cnnseg_encoders,
                        help="Encoder backbone to use for the encoder-decoder architecture (default: resnet50)")
    parser.add_argument("--pth_backbone_checkpoint", type=str, default=None,
                        help="Path to a checkpoint for the backbone encoder (default: None, so no checkpoint is loaded)")
    parser.add_argument("--bl_multiclass", action="store_true",
                        help="Whether to train in multiclass segmentation mode (default: False for binary)")
    parser.add_argument("--int_size", type=int, default=224, choices=[224, 518],
                        help="Spatial size for images and masks (default: 224)")

    parser.add_argument("--int_epochs", type=int, default=50,
                        help="Number of training epochs per phase (default: 50)")
    parser.add_argument("--int_patience", type=int, default=10,
                        help="Early stopping patience for training per phase (default: 10)")
    parser.add_argument("--int_batch", type=int, default=16,
                        help="Batch size for training per phase (default: 16)")

    parser.add_argument("--int_epochs_freeze", type=int, default=10,
                        help="Number of epochs to keep the backbone frozen before unfreezing it for fine-tuning (default: 10)")
    parser.add_argument("--bl_no_shuffle", action="store_true",
                        help="Whether to shuffle the training dataloader (default: False, so shuffling is enabled).")
    parser.add_argument("--bl_no_weighted_sampler", action="store_true",
                        help="Whether to use the weighted sampler for the training dataloader (default: False, so weighted sampler is enabled).")

    parser.add_argument("--flt_lr", type=float, default=1e-3,
                        help="Learning rate for training both supervised and semi-supervised models (default: 1e-3)")

    parser.add_argument("--int_workers", type=int, default=4,
                        help="Number of workers for classifier data loading (default: 4)")
    parser.add_argument("--int_devices", type=int, default=1,
                        help="Number of GPUs to train on (default: 1), reduce --int_batch accordingly.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main training entry point for segmentation."""
    logger.info("=" * 70)
    logger.info("LOADING TRAINING CONFIGURATION...")
    args: argparse.Namespace = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    utils.get_device()
    if utils.is_rank_zero():
        int_model: int = utils.record_model_parameters(dct_parameters=vars(args), bl_record=True)
    else:
        time.sleep(5) # allow rank 0 time to write to ledger
        int_model: int = utils.record_model_parameters(dct_parameters=vars(args), bl_record=False)
    str_class: str = "multi" if args.bl_multiclass else "binary"

    logger.info("=" * 70)
    logger.info("LOADING SUPERVISED DATA...")
    int_classes: int = utils.Instruments.int_classes if args.bl_multiclass else 2
    dct_dataloaders: dict[str, DataLoader] = initialise_dataloaders(
        str_task="seg",
        bl_multiclass=args.bl_multiclass,
        int_batch=args.int_batch,
        int_workers=args.int_workers,
        bl_shuffle=not args.bl_no_shuffle,
        bl_weighted_sampler=not args.bl_no_weighted_sampler,
        int_size=args.int_size,
    )

    logger.info("=" * 70)
    logger.info(f"TRAINING SUPERVISED MODEL...")
    model: L.LightningModule = CNNSegmenter(
        str_architecture=args.str_architecture,
        str_encoder=args.str_encoder,
        int_classes=int_classes,
        flt_lr=args.flt_lr,
        int_epochs_freeze=args.int_epochs_freeze,
        pth_backbone_checkpoint=args.pth_backbone_checkpoint if args.pth_backbone_checkpoint else None
    )
    checkpoint = ModelCheckpoint(
        dirpath=utils.Directories.pth_models,
        filename=f"{int_model}_seg",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=False,
        enable_version_counter=False,
    )
    trainer: L.Trainer = L.Trainer(
        max_epochs=args.int_epochs,
        callbacks=[checkpoint, EarlyStopping(monitor="val_dice", mode="max", patience=args.int_patience)],
        accelerator="auto",
        devices=args.int_devices,
        precision="16-mixed",
        strategy="ddp_find_unused_parameters_true" if args.int_epochs_freeze > 0 else "ddp",
        logger=TensorBoardLogger(save_dir=utils.Directories.pth_logs, name=str(int_model), sub_dir="seg"),
        enable_model_summary=False,
    )
    trainer.fit(model=model, train_dataloaders=dct_dataloaders["train"], val_dataloaders=dct_dataloaders["val"])
    
    logger.info("=" * 70)
    logger.info("EVALUATING SUPERVISED RESULTS...")
    for str_split, dataloaders in dct_dataloaders.items():
        results = trainer.test(model=model, ckpt_path=checkpoint.best_model_path, dataloaders=dataloaders, verbose=False)
        utils.save_results(
            results=results,
            int_model=int_model,
            str_model="segmentation",
            str_class=str_class,
            str_split=str_split,
            pth_results=utils.Directories.pth_results_seg,
        )
        logger.info(f"Results for {str_split} split:")
        for _, dataloader_results in enumerate(results):
            for key, value in dataloader_results.items():
                logger.info(f"{key}: {value}")

    logger.info("TRAINING COMPLETE!!!")
    logger.info("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
