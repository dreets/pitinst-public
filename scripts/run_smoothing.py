"""
Evaluate temporal smoothing on existing multiclass segmentation and classifier models.

Choice to run smoothing on classifier-only, CNN segmenter + classifier, or mmseg segmenter + classifier checkpoints.
Choice to run online smoothing (default) or offline smoothing (set --bl_offline).
Choice of temporal smoothing window size (default: 5 frames, set --int_clip).

Paths to csvs are found under data/segmentation/<model_number>/<video_name> for both classifier and segmenter results.
The default has no smoothing applied, so this script can be used to evaluate existing model outputs.

Results are saved to data/results/segmentation.csv, with entries of the form:
<segmentation_model_number>_seg+<classifier_model_number>_clf+smooth<int_clip>[offline|online].csv

To execute this, run:
    python scripts/run_smoothing.py --int_clf <classifier_model_number> --int_seg <segmentation_model_number> \
		[--int_clip <temporal_smoothing_window_size>] [--bl_offline]
For example:
	--int_clf 1 --int_seg 1
	--int_clf 1 --int_seg 2
	--int_clf 1 --int_seg 1 --int_clip 5
	--int_clf 1 --int_seg 1 --int_clip 7 --bl_offline
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from torchmetrics.segmentation import DiceScore

import utils

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate checkpoints with segmentation-weighted temporal smoothing.")
	parser.add_argument("--int_clf", type=int, required=True,
						help="Model number for saved annotations found in results, used for classification.")
	parser.add_argument("--int_seg", type=int, required=True,
						help="Model number for saved annotations found in results, used for segmentation.")

	parser.add_argument("--int_clip", type=int, default=0,
						help="Number of frames to aggregate over for temporal smoothing (default: 0).")
	parser.add_argument("--bl_offline", action="store_true",
						help="Offline uses future frames for smoothing, online uses only past frames (default: online).")
	return parser


def main(argv: list[str] | None = None) -> None:
	args = build_parser().parse_args(argv)
	for key, value in vars(args).items():
		logger.info(f"  {key}: {value}")

	for str_split in ["val", "test"]:
		int_classes: int = utils.Instruments.int_classes
		metrics: MetricCollection = MetricCollection({
			"f1_binary": F1Score(task="binary", average="macro"),
			"accuracy": Accuracy(task="multiclass", num_classes=int_classes, average="micro"),
			"precision": Precision(task="multiclass", num_classes=int_classes, average="macro"),
			"recall": Recall(task="multiclass", num_classes=int_classes, average="macro"),
			"f1_macro": F1Score(task="multiclass", num_classes=int_classes, average="macro"),
			"f1": F1Score(task="multiclass", num_classes=int_classes, average="macro", ignore_index=0),
			"dice": DiceScore(num_classes=int_classes, average="macro", include_background=False, input_format="index"),
			"dice_binary": DiceScore(num_classes=2, average="macro", include_background=True, input_format="index"),
			"f1_macro_per_class": F1Score(task="multiclass", num_classes=int_classes, average="none"),
		})
		
		lst_videos = utils.Videos.dct_split_videos[str_split]
		for str_video in lst_videos:		
			pth_results_clf: Path = utils.Directories.pth_segmentation / str(args.int_clf) / f"{str_video}.csv"
			pth_results_seg: Path = utils.Directories.pth_segmentation / str(args.int_seg) / f"{str_video}.csv"
			pth_results_gts: Path = utils.Directories.pth_annotations / f"{str_video}.csv"
			if not pth_results_clf.is_file() and not pth_results_seg.is_file() and not pth_results_gts.is_file():
				logger.warning(f"Skipping {str_video}: No results found for either classifier, segmenter, or ground truth.")
				continue

			int_chunk_size: int = 50
			df_clf_iterator = pd.read_csv(pth_results_clf, chunksize=int_chunk_size)
			df_seg_iterator = pd.read_csv(pth_results_seg, chunksize=int_chunk_size)
			df_gts_iterator = pd.read_csv(pth_results_gts, chunksize=int_chunk_size)
			
			lst_clf_preds_all: list[int] = []
			lst_clf_gts_all: list[int] = []
			lst_seg_preds_all: list[list[tuple[int, int]]] = []
			lst_seg_gts_all: list[list[tuple[int, int]]] = []
			for df_clf, df_seg, df_gts in zip(df_clf_iterator, df_seg_iterator, df_gts_iterator, strict=True):
				df_clf = df_clf.drop(columns=["LST_SEGMENTATION_PRED"])
				df_seg = df_seg.drop(columns=["STR_CLASS_PRED"])
				df_merged = df_clf.merge(df_seg, on="INT_FRAME", how="inner").merge(df_gts, on="INT_FRAME", how="inner")
				df_merged = df_merged.sort_values(by="INT_FRAME").reset_index(drop=True)
				del df_clf, df_seg, df_gts # free up memory

				for _, row in df_merged.iterrows():
					if str(row["STR_CLASS"]) not in utils.Instruments.dct_str_int_class:
						continue
					
					int_clf_pred: int = utils.Instruments.dct_str_int_class[str(row["STR_CLASS_PRED"])]
					int_clf_gt: int = utils.Instruments.dct_str_int_class[str(row["STR_CLASS"])]

					lst_coords_pred: list[float] = [float(coord) for coord in str(row["LST_SEGMENTATION_PRED"]).split(";")]
					lst_seg_pred: list[tuple[int, int]] = [(int(lst_coords_pred[i]), int(lst_coords_pred[i + 1])) for i in range(0, len(lst_coords_pred), 2)]

					lst_coords_gt: list[float] = [float(coord) for coord in str(row["LST_SEGMENTATION"]).split(";")]
					lst_seg_gt: list[tuple[int, int]] = [(int(lst_coords_gt[i]), int(lst_coords_gt[i + 1])) for i in range(0, len(lst_coords_gt), 2)]

					lst_clf_preds_all.append(int_clf_pred)
					lst_clf_gts_all.append(int_clf_gt)
					lst_seg_preds_all.append(lst_seg_pred)
					lst_seg_gts_all.append(lst_seg_gt)

				del df_merged # free up memory
				
			if args.int_clip > 0:
				lst_clf_preds_all, lst_seg_preds_all = temporal_smoothing(
					lst_clf_preds=lst_clf_preds_all,
					lst_seg_preds=lst_seg_preds_all,
					int_clip=args.int_clip, 
					bl_online=not args.bl_offline,
				)

			if lst_clf_preds_all:
				update_metrics_video(
					metrics=metrics,
					lst_clf_preds=lst_clf_preds_all,
					lst_clf_gts=lst_clf_gts_all,
					lst_seg_preds=lst_seg_preds_all,
					lst_seg_gts=lst_seg_gts_all,
					int_batch_size=50,
				)
				logger.info(f"Processed {len(lst_clf_preds_all)} frames for video {str_video}.")
			del lst_clf_preds_all, lst_clf_gts_all, lst_seg_preds_all, lst_seg_gts_all # free up memory
			del df_clf_iterator, df_seg_iterator, df_gts_iterator # free up memory

		str_model = f"{args.int_seg}_seg+{args.int_clf}_clf+smooth{args.int_clip}{'offline' if args.bl_offline else 'online'}"
		metrics_computed = metrics.compute()
		
		# Extract and print per-class f1 macro metrics (not saved to results, but useful for analysis)
		if "f1_macro_per_class" in metrics_computed:
			logger.info(f"\n{'='*70}")
			logger.info(f"F1 Macro per class for {str_model} ({str_split}):")
			logger.info(f"{'='*70}")
			for class_idx in range(int_classes):
				class_name = list(utils.Instruments.dct_str_int_class.keys())[class_idx]
				f1_value = float(metrics_computed["f1_macro_per_class"][class_idx])
				logger.info(f"  {class_name}: {f1_value:.4f}")
			logger.info(f"{'='*70}\n")
		
		results_dict = {k: float(v) if isinstance(v, torch.Tensor) else float(v) for k, v in metrics_computed.items() if k != "f1_macro_per_class"}
		utils.save_results(
			results=[results_dict],
			int_model=args.int_seg,
			str_model=str_model,
			str_class="multi",
			str_split=str_split,
			pth_results=utils.Directories.pth_results_seg,
		)


def update_metrics_video(
        metrics: MetricCollection,
        lst_clf_preds: list[int],
        lst_clf_gts: list[int],
        lst_seg_preds: list[list[tuple[int, int]]],
        lst_seg_gts: list[list[tuple[int, int]]],
        int_batch_size: int = 50,
    ) -> None:
    """
    Update metrics incrementally in batches to manage memory.
    Args:
        metrics: MetricCollection containing the metrics to update.
        lst_clf_preds: List of predicted class indices.
        lst_clf_gts: List of ground truth class indices.
        lst_seg_preds: List of predicted segmentation coordinates.
        lst_seg_gts: List of ground truth segmentation coordinates.
        int_batch_size: Number of frames to process at once before clearing intermediate data.
    Returns:
        None but updates the metrics in place.
    Raises:
        ValueError: If the lengths of the input lists do not match.
    """
    if not (len(lst_clf_preds) == len(lst_clf_gts) == len(lst_seg_preds) == len(lst_seg_gts)):
        raise ValueError("The lengths of the input lists do not match.")
    
    # Process in batches to avoid memory explosion
    for idx in range(0, len(lst_clf_preds), int_batch_size):
        int_end_idx = min(idx + int_batch_size, len(lst_clf_preds))
        
        lst_clf_preds_batch = lst_clf_preds[idx:int_end_idx]
        lst_clf_gts_batch = lst_clf_gts[idx:int_end_idx]
        lst_seg_preds_batch = lst_seg_preds[idx:int_end_idx]
        lst_seg_gts_batch = lst_seg_gts[idx:int_end_idx]
        
        tsr_preds_multi: torch.Tensor = torch.tensor(lst_clf_preds_batch, dtype=torch.long)
        tsr_gts_multi: torch.Tensor = torch.tensor(lst_clf_gts_batch, dtype=torch.long)
        
        metrics["f1_binary"].update((tsr_preds_multi > 0).long(), (tsr_gts_multi > 0).long())
        metrics["accuracy"].update(tsr_preds_multi, tsr_gts_multi)
        metrics["precision"].update(tsr_preds_multi, tsr_gts_multi)
        metrics["recall"].update(tsr_preds_multi, tsr_gts_multi)
        metrics["f1_macro"].update(tsr_preds_multi, tsr_gts_multi)
        metrics["f1"].update(tsr_preds_multi, tsr_gts_multi)
        metrics["f1_macro_per_class"].update(tsr_preds_multi, tsr_gts_multi)
        
        lst_mask_preds: list[np.ndarray] = []
        lst_mask_gts: list[np.ndarray] = []
        for pred_cls, pred_seg, gt_cls, gt_seg in zip(lst_clf_preds_batch, lst_seg_preds_batch, lst_clf_gts_batch, lst_seg_gts_batch):
            mask_pred = create_mask(lst_coords=pred_seg, int_class=pred_cls)
            mask_gt = create_mask(lst_coords=gt_seg, int_class=gt_cls)
            lst_mask_preds.append(mask_pred)
            lst_mask_gts.append(mask_gt)
        
        tsr_seg_preds: torch.Tensor = torch.tensor(np.array(lst_mask_preds), dtype=torch.long)
        tsr_seg_gts: torch.Tensor = torch.tensor(np.array(lst_mask_gts), dtype=torch.long)
        
        metrics["dice"].update(tsr_seg_preds, tsr_seg_gts)
        metrics["dice_binary"].update((tsr_seg_preds > 0).long(), (tsr_seg_gts > 0).long())
        
        del tsr_preds_multi, tsr_gts_multi, lst_mask_preds, lst_mask_gts, tsr_seg_preds, tsr_seg_gts # free up memory
        del lst_clf_preds_batch, lst_clf_gts_batch, lst_seg_preds_batch, lst_seg_gts_batch # free up memory


def create_mask(lst_coords: list[tuple[int, int]], int_class: int, int_width: int = 720, int_height: int = 720) -> np.ndarray:
    """
	Converts a list of (x,y) coordinates into a 2D pixel mask.
	Args:
		lst_coords: List of (x,y) coordinates representing the polygon.
		int_class: Class index to fill the polygon with.
		int_width: Width of the output mask.
		int_height: Height of the output mask.
	Returns:
		arr_mask: 2D numpy array of shape (height, int_width) with the polygon filled with int_class.
	Raises:
		ValueError: If polygon has fewer than 3 points (invalid polygon).
	"""
    arr_mask: np.ndarray = np.zeros((int_height, int_width), dtype=np.int32)
    
    if len(lst_coords) < 3:
        return arr_mask
    
    arr_pts: np.ndarray = np.array(lst_coords, dtype=np.int32)
    cv2.fillPoly(arr_mask, [arr_pts], int(int_class))
    return arr_mask


def temporal_smoothing(
		lst_clf_preds: list[int],
		lst_seg_preds: list[list[tuple[int, int]]],
		int_clip: int = 5,
		bl_online: bool = True
	) -> tuple[list[int], list[list[tuple[int, int]]]]:
	"""
	Smooths class predictions using a moving mode filter over a sliding window.
	Args:
		lst_clf_preds: List of predicted class indices.
		lst_seg_preds: List of predicted segmentations corresponding to each frame.
		int_clip: The number of frames to aggregate over (window size).
		bl_online: The smoothing mode to use.
			True: Aggregates from (current - int_clip + 1) up to the current frame.
			False: Aggregates int_clip // 2 frames before and after the current frame.
	Returns:
		lst_smoothed_clf_preds: List of smoothed class predictions (mode over window).
		lst_seg_preds: List of segmentations corresponding to the smoothed class predictions.
	Raises:
		ValueError: If int_clip is not positive.
	"""
	if int_clip <= 0:
		raise ValueError("int_clip must be positive.")
	
	lst_smoothed_clf_preds: list[int] = []
	for i in range(len(lst_clf_preds)):
		if bl_online:
			int_start_idx: int = max(0, i + 1 - int_clip)
			int_end_idx: int = i + 1
		else:
			int_clip_half: int = int_clip // 2
			int_start_idx: int = max(0, i - int_clip_half)
			int_end_idx: int = min(len(lst_clf_preds), i + int_clip_half + 1)
		
		lst_window: list[int] = lst_clf_preds[int_start_idx:int_end_idx]
		int_mode: int = max(
			set(lst_window),
			key=lambda x: (
				lst_window.count(x), # mode
				x == lst_clf_preds[i], # if tied, use the current frame's prediction if it is among the modes
				x != 0, # if stilltied, use the non-zero classes
				len(lst_window) - 1 - lst_window[::-1].index(x), # if still tied, prefer the most recent occurrence if tied
			)
		)
		lst_smoothed_clf_preds.append(int_mode)

	for i in range(len(lst_smoothed_clf_preds)):
		if lst_smoothed_clf_preds[i] == 0:
			lst_seg_preds[i] = [(0, 0)]
	
	return lst_smoothed_clf_preds, lst_seg_preds


if __name__ == "__main__":
	raise SystemExit(main())
