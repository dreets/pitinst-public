"""
Train a classification model.

Choice between fully-supervised (default) or semi-supervised modes (Flexmatch).
Choice between binary (default) or multiclass modes.
Choice of architecture: ResNet50 (default) or other supported architectures.
Choice to freeze the backbone for a number of epochs before unfreezing it for fine-tuning (default: no freezing).
Choice to apply dropout before the classifier head (default: no dropout).
Choice to add an LSTM on top of the classifier for temporal modeling (default: no LSTM).
Choice between distributed training (default) or single-GPU training (set --int_devices=1).
Choice between various other parameters (see argparse help for details).

Model paths are saved to data/models/<model_number>_clf.ckpt, and results are saved to data/results/classification.csv.
TensorBoard logs are saved to data/logs/<model_number>/.

To execute this, run:
    python scripts/train_classification.py --str_clf_backbone <backbone> [--bl_multiclass] [--int_size <image_size>] \
        [--pth_backbone_checkpoint <path_to_checkpoint>] [--int_epochs <num_epochs>] [--int_patience <early_stopping_patience>] \
        [--int_batch <batch_size>] [--int_epochs_freeze <num_epochs_freeze>] [--bl_no_dropout] [--bl_no_shuffle] \
        [--bl_no_weighted_sampler] [--flt_lr <learning_rate>] [--str_loss <loss_function>]
For example:
    --str_clf_backbone resnet50 --bl_multiclass --int_size 224 --int_epochs 50 --int_patience 5 --int_batch 64 --int_epochs_freeze 5 \
        --bl_no_dropout --bl_no_shuffle --bl_no_weighted_sampler --flt_lr 1e-4 --str_loss ce
"""
import argparse
import lightning as L
import time
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

import utils
from dataloaders import initialise_dataloaders, initialise_flexmatch_dataloaders
from models_classification import BaseClassifier, FlexMatchClassifier, LSTMClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a classification model.")

    parser.add_argument("--str_clf_backbone", type=str, default="resnet50", choices=utils.Models.lst_clf_backbones,
                        help="Backbone to use for the classification model (default: resnet50)")
    parser.add_argument("--bl_multiclass", action="store_true",
                        help="Whether to train in multiclass classification mode (default: False for binary)")
    parser.add_argument("--int_size", type=int, default=224, choices=[224, 518],
                        help="Size to which input images will be resized (default: 224)")
    parser.add_argument("--pth_backbone_checkpoint", type=str, default=None,
                        help="Path to a checkpoint for the backbone encoder (default: None, so no checkpoint is loaded)")

    parser.add_argument("--int_epochs", type=int, default=50,
                        help="Number of training epochs per phase (default: 50)")
    parser.add_argument("--int_patience", type=int, default=5,
                        help="Early stopping patience for training per phase (default: 5)")
    parser.add_argument("--int_batch", type=int, default=64,
                        help="Batch size for training per phase (default: 64)")

    parser.add_argument("--int_epochs_freeze", type=int, default=5,
                        help="Number of epochs to keep the backbone frozen before unfreezing it for fine-tuning (default: 5)")
    parser.add_argument("--bl_no_dropout", action="store_true",
                        help="Whether to apply dropout before the classifier head (default: False, so dropout is applied).")
    parser.add_argument("--bl_no_shuffle", action="store_true",
                        help="Whether to shuffle the training dataloader (default: False, so shuffling is enabled).")
    parser.add_argument("--bl_no_weighted_sampler", action="store_true",
                        help="Whether to use the weighted sampler for the training dataloader (default: False, so weighted sampler is enabled).")

    parser.add_argument("--flt_lr", type=float, default=1e-4,
                        help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--str_loss", type=str, default="ce", choices=["ce", "focal"],
                        help="Loss function for multiclass classification (default: ce for cross-entropy)")

    parser.add_argument("--bl_semisupervised", action="store_true",
                        help="Whether to enable semi-supervised training (default: False, so semi-supervised training is disabled).")
    parser.add_argument("--int_clip", type=int, default=0,
                        help="Number of consecutive frames to stack into a temporal clip for LSTM or temporal smoothing (default: 0, i.e. no temporal clips)")
    parser.add_argument("--bl_lstm", action="store_true",
                        help="Whether to train an LSTM on top of the classifier for temporal modeling (default: False, so no LSTM is trained).")

    parser.add_argument("--int_workers", type=int, default=8,
                        help="Number of workers for classifier data loading (default: 8)")
    parser.add_argument("--int_devices", type=int, default=1,
                        help="Number of GPUs to train on (default: 1), reduce --int_batch accordingly.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main training entry point for classification."""
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
    int_classes: int = utils.Instruments.int_classes if args.bl_multiclass else 2

    if args.pth_backbone_checkpoint is None:
        logger.info("=" * 70)
        logger.info(f"LOADING SUPERVISED DATA...")
        dct_dataloaders: dict[str, DataLoader] = initialise_dataloaders(
            str_task="clf",
            bl_multiclass=args.bl_multiclass,
            int_batch=args.int_batch,
            int_workers=args.int_workers,
            bl_shuffle=not args.bl_no_shuffle,
            bl_weighted_sampler=not args.bl_no_weighted_sampler,
            int_size=args.int_size,
        )

        logger.info("=" * 70)
        logger.info(f"TRAINING SUPERVISED MODEL...")
        clf_model: L.LightningModule = BaseClassifier(
            str_backbone=args.str_clf_backbone,
            flt_lr=args.flt_lr,
            int_classes=int_classes,
            bl_dropout=not args.bl_no_dropout,
            int_epochs_freeze=args.int_epochs_freeze,
            str_loss=args.str_loss,
        )
        clf_checkpoint = ModelCheckpoint(
            dirpath=utils.Directories.pth_models,
            filename=f"{int_model}_clf",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        )
        clf_trainer = L.Trainer(
            max_epochs=args.int_epochs,
            callbacks=[clf_checkpoint, EarlyStopping(monitor="val_f1", mode="max", patience=args.int_patience)],
            accelerator="auto",
            devices=args.int_devices,
            precision="16-mixed",
            strategy="ddp_find_unused_parameters_true" if args.int_epochs_freeze > 0 else "ddp",
            logger=TensorBoardLogger(save_dir=utils.Directories.pth_logs, name=str(int_model), sub_dir="clf"),
            enable_model_summary=False,
        )
        clf_trainer.fit(model=clf_model, train_dataloaders=dct_dataloaders["train"], val_dataloaders=dct_dataloaders["val"])

        logger.info("=" * 70)
        logger.info("EVALUATING SUPERVISED RESULTS...")
        for str_split, dataloaders in dct_dataloaders.items():
            results = clf_trainer.test(model=clf_model, ckpt_path=clf_checkpoint.best_model_path, dataloaders=dataloaders, verbose=False)
            utils.save_results(
                results=results,
                int_model=int_model,
                str_model="classifier",
                str_class=str_class,
                str_split=str_split,
                pth_results=utils.Directories.pth_results_clf,
            )
            logger.info(f"Results for {str_split} split:")
            for _, dataloader_results in enumerate(results):
                for key, value in dataloader_results.items():
                    logger.info(f"{key}: {value}")

    if args.bl_semisupervised:
        logger.info("=" * 70)
        logger.info("LOADING SEMI-SUPERVISED DATA...")
        dct_flexmatch_dataloaders: dict[str, DataLoader] = initialise_flexmatch_dataloaders(
            bl_multiclass=args.bl_multiclass,
            int_batch=args.int_batch // 4, # reduce batch size for FlexMatch to avoid OOM errors
            int_mu=int(7 * 224 / args.int_size), # scale mu by input size to keep the same number of pixels per batch
            int_workers=args.int_workers,
            int_size=args.int_size,
            bl_shuffle=not args.bl_no_shuffle,
            bl_weighted_sampler=not args.bl_no_weighted_sampler,
        )

        logger.info("=" * 70)
        logger.info(f"TRAINING FLEXMATCH SEMI-SUPERVISED MODEL...")
        flexmatch_model: L.LightningModule = FlexMatchClassifier(
            str_backbone=args.str_clf_backbone,
            int_unlabeled=len(dct_flexmatch_dataloaders["train_unlabeled"].dataset), # type: ignore
            int_classes=int_classes,
            flt_lr=args.flt_lr,
            flt_threshold=0.95,
            flt_lambda_u=1.0,
            int_epochs_freeze=args.int_epochs_freeze,
            bl_dropout=not args.bl_no_dropout,
        )
        str_backbone_checkpoint = args.pth_backbone_checkpoint if args.pth_backbone_checkpoint is not None else clf_checkpoint.best_model_path # type: ignore 
        logger.info(f"Warm-starting FlexMatch backbone from supervised checkpoint: {str_backbone_checkpoint}")
        dct_clf_state: dict = torch.load(str_backbone_checkpoint, map_location="cpu")["state_dict"] # type: ignore 
        dct_backbone_state: dict = {key: value for key, value in dct_clf_state.items() if key.startswith("backbone.")}
        flexmatch_model.load_state_dict(dct_backbone_state, strict=False)
        flexmatch_checkpoint = ModelCheckpoint(
            dirpath=utils.Directories.pth_models,
            filename=f"{int_model}_flex",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        )
        flexmatch_trainer = L.Trainer(
            max_epochs=args.int_epochs,
            callbacks=[flexmatch_checkpoint, EarlyStopping(monitor="val_f1", mode="max", patience=args.int_patience)],
            accelerator="auto",
            devices=args.int_devices,
            precision="16-mixed",
            strategy="ddp_find_unused_parameters_true",
            logger=TensorBoardLogger(save_dir=utils.Directories.pth_logs, name=str(int_model), sub_dir="flexmatch"),
            enable_model_summary=False,
        )
        flexmatch_trainer.fit(
            model=flexmatch_model,
            train_dataloaders=CombinedLoader(
                {"labeled": dct_flexmatch_dataloaders["train"], "unlabeled": dct_flexmatch_dataloaders["train_unlabeled"]},
                mode="max_size_cycle",
            ),
            val_dataloaders=dct_flexmatch_dataloaders["val"],
        )

        logger.info("=" * 70)
        logger.info("EVALUATING FLEXMATCH RESULTS...")
        for str_split in ["train", "val", "test"]:
            results = flexmatch_trainer.test(
                model=flexmatch_model,
                ckpt_path=flexmatch_checkpoint.best_model_path,
                dataloaders=dct_flexmatch_dataloaders[str_split],
                verbose=False,
            )
            utils.save_results(
                results=results,
                int_model=int_model,
                str_model="flexmatch",
                str_class=str_class,
                str_split=str_split,
                pth_results=utils.Directories.pth_results_clf,
            )
            logger.info(f"Results for {str_split} split:")
            for _, dataloader_results in enumerate(results):
                for key, value in dataloader_results.items():
                    logger.info(f"{key}: {value}")

    if args.bl_lstm and args.int_clip > 0:
        logger.info("=" * 70)
        logger.info(f"LOADING LSTM DATA...")
        dct_lstm_dataloaders: dict[str, DataLoader] = initialise_dataloaders(
            str_task="clf",
            bl_multiclass=args.bl_multiclass,
            int_batch=args.int_batch // args.int_clip,
            int_workers=args.int_workers,
            int_size=args.int_size,
            int_clip=args.int_clip,
        )

        logger.info(f"TRAINING LSTM MODEL...")
        logger.info("=" * 70)
        str_backbone_checkpoint = args.pth_backbone_checkpoint if args.pth_backbone_checkpoint is not None else clf_checkpoint.best_model_path # type: ignore
        clf_model: L.LightningModule = BaseClassifier.load_from_checkpoint(str_backbone_checkpoint)
        lstm_model = LSTMClassifier(pretrained_model=clf_model, flt_lr=args.flt_lr)
        lstm_checkpoint = ModelCheckpoint(
            dirpath=utils.Directories.pth_models,
            filename=f"{int_model}_lstm",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_last=False,
            enable_version_counter=False,
        )
        lstm_trainer = L.Trainer(
            max_epochs=args.int_epochs,
            callbacks=[lstm_checkpoint, EarlyStopping(monitor="val_f1", mode="max", patience=args.int_patience)],
            accelerator="auto",
            devices=args.int_devices,
            strategy="ddp_find_unused_parameters_true",
            logger=TensorBoardLogger(save_dir=utils.Directories.pth_logs, name=str(int_model), sub_dir="lstm"),
            enable_model_summary=False,
        )
        lstm_trainer.fit(model=lstm_model, train_dataloaders=dct_lstm_dataloaders["train"], val_dataloaders=dct_lstm_dataloaders["val"])

        logger.info("=" * 70)
        logger.info("EVALUATING LSTM RESULTS...")
        for str_split, dataloaders in dct_lstm_dataloaders.items():
            results = lstm_trainer.test(model=lstm_model, ckpt_path=lstm_checkpoint.best_model_path, dataloaders=dataloaders, verbose=False)
            utils.save_results(
                results=results,
                int_model=int_model,
                str_model="lstm",
                str_class=str_class,
                str_split=str_split,
                pth_results=utils.Directories.pth_results_clf,
            )
            logger.info(f"Results for {str_split} split:")
            for _, dataloader_results in enumerate(results):
                for key, value in dataloader_results.items():
                    logger.info(f"{key}: {value}")

    logger.info("TRAINING COMPLETE!!!")
    logger.info("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
