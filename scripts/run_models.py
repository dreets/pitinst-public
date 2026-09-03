"""
Run models (loaded from their checkpoints) and save the results to a CSV file.
Results are saved to data/results/<model_number>/<video_name>.csv.

To execute this, run:
    python scripts/run_models.py --str_evaluator <evaluator> --pth_model <model_checkpoint> \
        [--bl_binary] [--int_size <size>] [--str_splits <splits>] \
        [--str_mmseg_architecture <mmseg_architecture>] [--str_mmseg_encoder <mmseg_encoder>]
For example:
    --str_evaluator clf --pth_model models/1_clf.ckpt --int_size 518
    --str_evaluator flex --pth_model models/1_flex.ckpt --int_size 518
    --str_evaluator cnnseg --pth_model models/1_seg.ckpt --bl_binary --int_size 224 --str_splits all
    --str_evaluator mmseg --pth_model models/1_seg.pth --int_size 518 --str_mmseg_architecture mask2former --str_mmseg_encoder dinov2
    --str_evaluator lstm --pth_model models/1_lstm.ckpt --pth_lstm_backbone models/1_lstm_backbone.ckpt --bl_multiclass --int_size 518 --int_clip 5
"""
import argparse
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import utils
from dataloaders import initialise_video_dataloaders
from models_classification import BaseClassifier, FlexMatchClassifier, LSTMClassifier
from models_segmentation import CNNSegmenter, SegInitialisers, initialise_mmseg

import mmcv
_str_mmcv_version: str = mmcv.__version__
mmcv.__version__ = "2.1.0"
import mmseg.models # noqa: F401
from mmengine.config import ConfigDict
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.structures import SegDataSample
mmcv.__version__ = _str_mmcv_version

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run models and save predictions to CSV.")

    parser.add_argument("--str_evaluator", choices=utils.Models.lst_evaluators,
                        help="The type of model which is being evaluated.")
    parser.add_argument("--pth_model", type=Path, default=None,
                        help="Path to classifier checkpoint, of the form models/<int_model>_clf.ckpt.")

    parser.add_argument("--bl_binary", action="store_true",
                        help="Disable multiclass classification (default: True for multiclass classification).")
    parser.add_argument("--int_size", type=int, default=224, choices=[224, 518],
                        help="Input image size used by the model (default: 224, 518 is for dinov2).")
    parser.add_argument("--str_splits", choices=utils.Videos.dct_split_videos.keys(), default="val_test",
                        help="Video split to evaluate on (default: val_test).")

    parser.add_argument("--str_mmseg_architecture", choices=utils.Models.lst_mmseg_architectures,
                        help="mmseg decoder architecture, required for str_evaluator=mmseg (default: mask2former).")
    parser.add_argument("--str_mmseg_encoder", choices=utils.Models.lst_mmseg_encoders,
                        help="mmseg encoder/classifier architecture, required for str_evaluator=mmseg (default: dinov2).")

    parser.add_argument("--pth_lstm_backbone", type=Path, default=None,
                        help="Path to pretrained backbone for LSTM model, required for str_evaluator=lstm.")
    parser.add_argument("--int_clip", type=int, default=5,
                        help="Number of consecutive frames, required for str_evaluator=lstm (default: 5).")

    return parser


def main(argv: list[str] | None = None) -> None:
    logger.info("=" * 70)
    logger.info("LOADING PARAMETERS...")
    args = build_parser().parse_args(argv)
    if args.pth_model is None or not args.pth_model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.pth_model}")
    if args.str_evaluator == "lstm" and (args.pth_lstm_backbone is None or not args.pth_lstm_backbone.is_file()):
        raise FileNotFoundError(f"LSTM backbone not found: {args.pth_lstm_backbone}")
    if args.str_evaluator == "mmseg" and (args.str_mmseg_architecture is None or args.str_mmseg_encoder is None):
        raise ValueError("For str_evaluator=mmseg, both --str_mmseg_architecture and --str_mmseg_encoder must be specified.")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    int_model: int = int(args.pth_model.stem.split("_")[0])
    pth_results_seg: Path = utils.Directories.pth_segmentation / str(int_model)
    pth_results_seg.mkdir(parents=True, exist_ok=True)
    dct_int_str_class: dict[int, str] = utils.Instruments.dct_int_str_class
    ts_device: torch.device = utils.get_device()

    logger.info("=" * 70)
    logger.info("LOADING DATA...")

    dct_videos_dataloaders = initialise_video_dataloaders(
        str_task="clf" if args.str_evaluator == "clf" else "seg",
        bl_multiclass=not args.bl_binary,
        int_size=args.int_size,
        int_batch=16,
        int_workers=4,
        lst_videos=utils.Videos.dct_split_videos[args.str_splits],
    )

    logger.info("=" * 70)
    logger.info("RUNNING MODEL...")
    if args.str_evaluator in ["clf", "flex"]:
        if args.str_evaluator == "flex":
            classifier_model: L.LightningModule = FlexMatchClassifier.load_from_checkpoint(args.pth_model).to(ts_device).eval()
        else:
            classifier_model: L.LightningModule = BaseClassifier.load_from_checkpoint(args.pth_model).to(ts_device).eval()

        for str_video, dataloader in dct_videos_dataloaders.items():
            lst_predictions: list[dict] = []
            dataset = dataloader.dataset
            
            for int_batch_idx, batch in enumerate(dataloader):
                dct_outputs: dict[str, torch.Tensor] = classifier_model.predict_step(
                    batch=(batch[0].to(ts_device), batch[1].to(ts_device)), 
                    batch_idx=int_batch_idx,
                )

                int_start_idx: int = int_batch_idx * batch[0].size(0)
                for int_pred_idx, tsr_pred in enumerate(dct_outputs["tsr_preds_multi"]):
                    int_frame_idx: int = int_start_idx + int_pred_idx
                    int_pred: int = int(tsr_pred.item())
                    if int_frame_idx < len(dataset): # type: ignore
                        pth_frame, _int_class, _lst_polygon = dataset.lst_frames[int_frame_idx] # type: ignore
                        lst_predictions.append({
                            "STR_VIDEO": str_video,
                            "INT_FRAME": int(pth_frame.stem),
                            "STR_CLASS_PRED": dct_int_str_class[int_pred] if not args.bl_binary else ("unclassified" if int_pred == 1 else "no_instrument"),
                            "LST_SEGMENTATION_PRED": "0;0", # no segmentation for classifier
                        })
            
            save_predictions_to_csv(lst_predictions=lst_predictions, pth_results=pth_results_seg, str_video=str_video)

    if args.str_evaluator == "cnnseg":
        segmenter_model: L.LightningModule = CNNSegmenter.load_from_checkpoint(args.pth_model).to(ts_device).eval()
        for str_video, dataloader in dct_videos_dataloaders.items():
            lst_predictions: list[dict] = []
            dataset = dataloader.dataset
            
            for int_batch_idx, batch in enumerate(dataloader):
                dct_outputs: dict[str, torch.Tensor] = segmenter_model.predict_step(
                    batch=(batch[0].to(ts_device), batch[1].to(ts_device)), 
                    batch_idx=int_batch_idx
                )
                int_start_idx: int = int_batch_idx * batch[0].size(0)
                for int_pred_idx, (tsr_pred_mask, tsr_pred_class) in enumerate(zip(dct_outputs["tsr_preds_multi"], dct_outputs["tsr_preds_multi_clf"])):
                    int_frame_idx: int = int_start_idx + int_pred_idx
                    if int_frame_idx < len(dataset): # type: ignore
                        pth_frame, int_class, lst_polygon = dataset.lst_frames[int_frame_idx] # type: ignore

                        # convert predicted mask to numpy array and resize to 720p
                        npy_mask: np.ndarray = tsr_pred_mask.cpu().numpy().squeeze()
                        lst_coords: list[int] = extract_contours_from_mask(npy_mask)
                        str_segmentation_preds: str = ";".join(str(c) for c in lst_coords)

                        lst_predictions.append({
                            "STR_VIDEO": str_video,
                            "INT_FRAME": int(pth_frame.stem),
                            "STR_CLASS_PRED": dct_int_str_class[int(tsr_pred_class)] if not args.bl_binary else ("unclassified" if int(tsr_pred_class) == 1 else "no_instrument"),
                            "LST_SEGMENTATION_PRED": str_segmentation_preds,
                        })
            
            save_predictions_to_csv(lst_predictions=lst_predictions, pth_results=pth_results_seg, str_video=str_video)

    if args.str_evaluator == "mmseg":
        register_all_modules(init_default_scope=True)
        dct_config = dict(
            type="EncoderDecoder",
            data_preprocessor=dict(type="SegDataPreProcessor", mean=None, std=None, bgr_to_rgb=False, pad_val=0, seg_pad_val=255, size=(args.int_size, args.int_size)),
            backbone=SegInitialisers.backbone_mmseg(str_backbone=args.str_mmseg_encoder, str_backbone_checkpoint=None),
            neck=SegInitialisers.neck_mmseg(str_architecture=args.str_mmseg_architecture),
            decode_head=initialise_mmseg(str_architecture=args.str_mmseg_architecture, int_classes=utils.Instruments.int_classes),
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        )
        
        runner = Runner(
            model=ConfigDict(dct_config),
            default_scope="mmseg",
            cfg=ConfigDict(dict(type="EpochBasedTrainLoop", max_epochs=1)),
            work_dir=str(utils.Directories.pth_logs / str(int_model)),
        )
        runner.load_checkpoint(str(args.pth_model))
        runner.model.eval()
        
        for str_video, dataloader in dct_videos_dataloaders.items():
            lst_predictions: list[dict] = []
            dataset = dataloader.dataset
            
            with torch.no_grad():
                for int_batch_idx, data_batch in enumerate(dataloader):
                    if isinstance(data_batch, dict):
                        images = data_batch.get("inputs", data_batch.get("img", None))
                        images = images.to(ts_device) if isinstance(images, torch.Tensor) else images
                        int_batch_size = images.size(0) if isinstance(images, torch.Tensor) else len(images) # type: ignore
                        
                        lst_data_samples_input = []
                        img_h, img_w = images.shape[-2:] if images.ndim == 4 else (images.shape[-2], images.shape[-1]) # type: ignore
                        for i in range(int_batch_size):
                            data_sample = SegDataSample()
                            metainfo_dict = {"ori_shape": (img_h, img_w), "img_shape": (img_h, img_w), "pad_shape": (img_h, img_w), "scale_factor": 1.0}
                            if "data_samples" in data_batch and isinstance(data_batch["data_samples"], list):
                                if i < len(data_batch["data_samples"]):
                                    ds = data_batch["data_samples"][i]
                                    if hasattr(ds, 'metainfo') and isinstance(ds.metainfo, dict):
                                        metainfo_dict.update(ds.metainfo)
                            data_sample.set_metainfo(metainfo_dict)
                            lst_data_samples_input.append(data_sample)
                        data_batch = {"inputs": images, "data_samples": lst_data_samples_input}

                    else:
                        images, _labels = data_batch
                        images = images.to(ts_device) if isinstance(images, torch.Tensor) else images
                        int_batch_size = images.size(0) if isinstance(images, torch.Tensor) else len(images)
                        
                        lst_data_samples_input = []
                        img_h, img_w = images.shape[-2:] if images.ndim == 4 else (images.shape[-2], images.shape[-1])
                        for i in range(int_batch_size):
                            data_sample = SegDataSample()
                            data_sample.set_metainfo({"ori_shape": (img_h, img_w), "img_shape": (img_h, img_w), "pad_shape": (img_h, img_w), "scale_factor": 1.0})
                            lst_data_samples_input.append(data_sample)
                        data_batch = {"inputs": images, "data_samples": lst_data_samples_input}
                    
                    lst_data_samples: list = runner.model.val_step(data_batch)
                    
                    int_start_idx: int = int_batch_idx * int_batch_size
                    for int_pred_idx, data_sample in enumerate(lst_data_samples):
                        int_frame_idx: int = int_start_idx + int_pred_idx
                        if int_frame_idx < len(dataset): # type: ignore
                            pth_frame, int_class, lst_polygon = dataset.lst_frames[int_frame_idx] # type: ignore

                            # convert predicted mask to numpy array and resize to 720p
                            tsr_pred_mask: torch.Tensor = data_sample.pred_sem_seg.data.squeeze(0)
                            npy_mask: np.ndarray = tsr_pred_mask.cpu().numpy().squeeze()
                            lst_coords: list[int] = extract_contours_from_mask(npy_mask)
                            str_segmentation_preds: str = ";".join(str(c) for c in lst_coords)

                            tsr_pred_flat: torch.Tensor  = tsr_pred_mask.flatten()
                            tsr_pred_nonzero: torch.Tensor = tsr_pred_flat[tsr_pred_flat != 0]
                            if tsr_pred_nonzero.numel() > 0:
                                int_pred_class: int = int(tsr_pred_nonzero.mode().values.item())
                            else:
                                int_pred_class: int = 0
                            
                            lst_predictions.append({
                                "STR_VIDEO": str_video,
                                "INT_FRAME": int(pth_frame.stem),
                                "STR_CLASS_PRED": dct_int_str_class[int_pred_class],
                                "LST_SEGMENTATION_PRED": str_segmentation_preds,
                            })

            save_predictions_to_csv(lst_predictions=lst_predictions, pth_results=pth_results_seg, str_video=str_video)

    if args.str_evaluator == "lstm":
        lstm_backbone: L.LightningModule = BaseClassifier.load_from_checkpoint(args.pth_lstm_backbone).to(ts_device).eval()
        lstm_model: L.LightningModule = LSTMClassifier.load_from_checkpoint(args.pth_model, pretrained_model=lstm_backbone).to(ts_device).eval()
        
        for str_video, dataloader in dct_videos_dataloaders.items():
            lst_predictions: list[dict] = []
            dataset = dataloader.dataset # type: ignore
            
            lst_all_frames = []
            for int_batch_idx, batch in enumerate(dataloader):
                for int_frame_idx in range(batch[0].size(0)):
                    lst_all_frames.append((batch[0][int_frame_idx], batch[1][int_frame_idx]))
            
            # Process frames with temporal context (sliding window of consecutive frames)
            for int_frame_idx in range(len(lst_all_frames)):
                int_start_clip = max(0, int_frame_idx - args.int_clip + 1)
                lst_clip_frames = lst_all_frames[int_start_clip:int_frame_idx + 1]
                while len(lst_clip_frames) < args.int_clip:
                    lst_clip_frames.insert(0, lst_clip_frames[0])
                
                tsr_clip = torch.stack([frame[0] for frame in lst_clip_frames]).unsqueeze(0)  # [1, int_clip, 3, H, W]
                tsr_label = lst_clip_frames[-1][1].unsqueeze(0) # [1]
                
                with torch.no_grad():
                    dct_outputs: dict[str, torch.Tensor] = lstm_model.predict_step(
                        batch=(tsr_clip.to(ts_device), tsr_label.to(ts_device)),
                        batch_idx=0
                    )
                
                int_pred: int = int(dct_outputs["tsr_preds_multi"][0].item())
                pth_frame, _int_class, _lst_polygon = dataset.lst_frames[int_frame_idx] # type: ignore
                lst_predictions.append({
                    "STR_VIDEO": str_video,
                    "INT_FRAME": int(pth_frame.stem),
                    "STR_CLASS_PRED": dct_int_str_class[int_pred] if not args.bl_binary else ("unclassified" if int_pred == 1 else "no_instrument"),
                    "LST_SEGMENTATION_PRED": "0;0", # no segmentation for lstm classifier
                })
            
            save_predictions_to_csv(lst_predictions=lst_predictions, pth_results=pth_results_seg, str_video=str_video)

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETED!")


def save_predictions_to_csv(lst_predictions: list[dict], pth_results: Path, str_video: str) -> None:
    if lst_predictions:
        df_predictions: pd.DataFrame = pd.DataFrame(lst_predictions)
        pth_output: Path = pth_results / f"{str_video}.csv"
        df_predictions.to_csv(pth_output, index=False)
        logger.info(f"Saved {len(lst_predictions)} predictions for video '{str_video}' to {pth_output}")
    else:
        logger.warning(f"No predictions found for video '{str_video}'")


def extract_contours_from_mask(npy_mask: np.ndarray) -> list[int]:
    npy_mask_binary = (npy_mask > 0).astype(np.uint8)
    if npy_mask_binary.sum() == 0:
        return [0, 0]
    
    contours, _ = cv2.findContours(npy_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0]

    lst_coords: list[int] = []
    largest_contour = max(contours, key=cv2.contourArea)
    for point in largest_contour:
        x, y = point[0] # type: ignore
        lst_coords.extend([int(x / npy_mask.shape[1] * 720), int(y / npy_mask.shape[0] * 720)])
    
    return lst_coords


if __name__ == "__main__":
	raise SystemExit(main())