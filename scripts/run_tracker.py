"""
Run SAM2 tracker to propagate segmentation masks across video frames.
This script takes 1 int_fps segmentation predictions (frame ID, class ID, str_coordinates mask) and 
uses the SAM2 (Segment Anything 2) video predictor to interpolate masks for all intermediate frames (between keyframes).
Applies offline temporal smoothing to class predictions before tracking.

To execute this script, run:
    python scripts/run_tracker.py --int_seg <int_seg> --str_video <str_video> [--int_clip <int_clip>] 
    [--sam2checkpoint <sam2checkpoint>] [--sam2cfg <sam2cfg>]
For example:
    python scripts/run_tracker.py --int_seg 1 --str_video data/videos/Antimony.mp4
    python scripts/run_tracker.py --int_seg 1 --str_video data/videos/Antimony.mp4 --int_clip 5 --sam2checkpoint models/sam2_hiera_large.pt --sam2cfg sam2_hiera_l.yaml
"""
import argparse
import csv
import cv2
import os
import numpy as np
import pandas as pd

import torch
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

import run_smoothing
import utils

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# supress tqdm output for non-keyframe chunks (built into SAM2)
from tqdm import tqdm
_orig_tqdm_init = tqdm.__init__
def _quiet_tqdm_init(self, *args, **kwargs):
    if kwargs.get("desc") != "Tracking keyframe chunks":
        kwargs["disable"] = True
    _orig_tqdm_init(self, *args, **kwargs)
tqdm.__init__ = _quiet_tqdm_init


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM2 tracker over segmentation outputs.")

    parser.add_argument("--int_seg", required=True, type=str,
                        help="Model number for saved annotations found in results.")
    parser.add_argument("--str_video", required=True, type=str,
                        help="Video for which you want to run tracking.")
    parser.add_argument("--int_clip", type=int, default=7,
                        help="Number of seconds to aggregate over for temporal smoothing (default: 7).")
    parser.add_argument("--str_sam2model", type=str, choices=["large", "tiny"], default="tiny",
                        help="SAM 2 model to use (default: 'tiny'), tiny is 2-3x faster but less accurate.")

    return parser


def main(argv: list[str] | None = None) -> None:
    logger.info("=" * 70)
    logger.info("LOADING PARAMETERS...")
    args = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    pth_results_seg: Path = utils.Directories.pth_segmentation/ str(args.int_seg) / f"{Path(args.str_video)}.csv"
    pth_results_tracked: Path = utils.Directories.pth_tracking / str(args.int_seg) / f"{Path(args.str_video)}.csv"
    pth_results_tracked.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEFRAMING VIDEO...")
    pth_frames: Path = deframe_video(str_video=args.str_video)

    logger.info("=" * 70)
    logger.info("LOADING SEGMENTATIONS & SMOOTHING...")
    lst_frames: list[int] = []
    lst_clf: list[int] = []
    lst_seg: list[list[tuple[int, int]]] = []
    df_seg = pd.read_csv(pth_results_seg)
    for _, row in df_seg.iterrows():
        if str(row["STR_CLASS_PRED"]) not in utils.Instruments.dct_str_int_class:
            continue

        lst_coords_pred: list[float] = [float(coord) for coord in str(row["LST_SEGMENTATION_PRED"]).split(";")]
        lst_seg_pred: list[tuple[int, int]] = [(int(lst_coords_pred[i]), int(lst_coords_pred[i + 1])) for i in range(0, len(lst_coords_pred), 2)]

        lst_frames.append(int(row["INT_FRAME"]))
        lst_clf.append(utils.Instruments.dct_str_int_class[str(row["STR_CLASS_PRED"])])
        lst_seg.append(lst_seg_pred)
    del df_seg  # free up memory

    if args.int_clip > 0:
        lst_clf, lst_seg = run_smoothing.temporal_smoothing(lst_clf_preds=lst_clf, lst_seg_preds=lst_seg, int_clip=args.int_clip, bl_online=False)

    lst_keyframes: list[tuple[int, int, list[tuple[int, int]]]] = sorted(zip(lst_frames, lst_clf, lst_seg, strict=True), key=lambda x: x[0])
    dct_keyframes: dict[int, tuple[int, list[tuple[int, int]]]] = {f: (c, s) for f, c, s in lst_keyframes}
    del lst_frames, lst_clf, lst_seg # free up memory

    logger.info("=" * 70)
    logger.info("TRACKING WITH SAM2...")
    pth_chunk_temp = pth_frames.parent / f"{pth_frames.name}_chunks"
    pth_chunk_temp.mkdir(parents=True, exist_ok=True)

    with (
        utils.inference_context() as device,
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        open(pth_results_tracked, mode="w", newline="", encoding="utf-8") as f_out,
    ):
        writer = csv.writer(f_out)
        writer.writerow(["INT_FRAME", "STR_CLASS_PRED", "LST_SEGMENTATION_PRED"])

        if args.str_sam2model == "large":
            args.sam2cfg = "sam2_hiera_l.yaml"
            args.sam2checkpoint = "models/sam2_hiera_large.pt"
        elif args.str_sam2model == "tiny":
            args.sam2cfg = "sam2_hiera_t.yaml"
            args.sam2checkpoint = "models/sam2_hiera_tiny.pt"
        else:
            raise ValueError(f"Unsupported SAM 2 model: {args.str_sam2model}")

        predictor: SAM2VideoPredictor = build_sam2_video_predictor(args.sam2cfg, args.sam2checkpoint, device=device) # type: ignore

        for idx in tqdm(range(len(lst_keyframes) - 1), desc="Tracking keyframe chunks", unit="chunk"):
            int_frame_start, int_class_start, lst_seg_start = lst_keyframes[idx]
            int_frame_end, int_class_end, lst_seg_end = lst_keyframes[idx + 1]

            str_class_start = utils.Instruments.dct_int_str_class[int_class_start]
            writer.writerow([int_frame_start, str_class_start, ";".join(f"{c};{d}" for c, d in lst_seg_start)])

            int_gap: int = int_frame_end - int_frame_start - 1
            if int_gap <= 0:
                continue

            bl_has_start_mask: bool = len(lst_seg_start) >= 3 and int_class_start != 0
            bl_has_end_mask: bool = len(lst_seg_end) >= 3 and int_class_end != 0

            if not bl_has_start_mask and not bl_has_end_mask:
                str_class_empty = utils.Instruments.dct_int_str_class[0]
                for int_out_frame in range(int_frame_start + 1, int_frame_end):
                    writer.writerow([int_out_frame, str_class_empty, "0;0"])
                continue

            for f in pth_chunk_temp.glob("*.jpg"):
                f.unlink()

            for int_new_idx, int_old_idx in enumerate(range(int_frame_start, int_frame_end + 1)):
                pth_src_img: Path = pth_frames / f"{int_old_idx:06d}.jpg"
                pth_dst_symlink: Path = pth_chunk_temp / f"{int_new_idx:05d}.jpg"
                if pth_src_img.exists():
                    os.symlink(pth_src_img, pth_dst_symlink)

            inference_state = predictor.init_state(video_path=str(pth_chunk_temp), async_loading_frames=False)

            if bl_has_start_mask:
                arr_mask_a: np.ndarray = run_smoothing.create_mask(lst_seg_start, int_class=1, int_width=720, int_height=720)
                if arr_mask_a.any():
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=int_class_start,
                        mask=arr_mask_a,
                    )

            if bl_has_end_mask:
                arr_mask_b: np.ndarray = run_smoothing.create_mask(lst_seg_end, int_class=1, int_width=720, int_height=720)
                if arr_mask_b.any():
                    rel_end_idx = int_frame_end - int_frame_start
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=rel_end_idx,
                        obj_id=int_class_end,
                        mask=arr_mask_b,
                    )

            predictor_propogate = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=1,
                max_frame_num_to_track=int_gap,
            )
            for (int_rel_out_frame, obj_out_ids, mask_out_logits) in predictor_propogate:
                int_out_frame: int = int_frame_start + int_rel_out_frame

                if int_out_frame in dct_keyframes:
                    int_out_clf, lst_out_seg = dct_keyframes[int_out_frame]
                else:
                    int_out_clf: int = 0
                    lst_out_seg: list[tuple[int, int]] = [(0, 0)]
                    flt_best_confidence: float = 0.0

                    for obj_id, mask_logit in zip(obj_out_ids, mask_out_logits):
                        tsr_foreground: torch.Tensor = mask_logit[0] > 0.0
                        if not tsr_foreground.any():
                            continue
                        flt_confidence: float = (mask_logit[0][tsr_foreground].mean().item())
                        if flt_confidence > flt_best_confidence:
                            arr_binary_mask = tsr_foreground.cpu().numpy().astype(np.uint8)
                            lst_contours, _ = cv2.findContours(arr_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if lst_contours:
                                arr_largest: np.ndarray = max(lst_contours, key=cv2.contourArea)
                                arr_coords: np.ndarray = arr_largest.reshape(-1, 2)
                                int_out_clf = obj_id
                                lst_out_seg = [(int(pt[0]), int(pt[1])) for pt in arr_coords]
                                flt_best_confidence = flt_confidence

                writer.writerow([int_out_frame, utils.Instruments.dct_int_str_class[int_out_clf], ";".join(f"{c};{d}" for c, d in lst_out_seg)])

        int_frame_last, int_class_last, lst_seg_last = lst_keyframes[-1]
        writer.writerow([int_frame_last, utils.Instruments.dct_int_str_class[int_class_last], ";".join(f"{c};{d}" for c, d in lst_seg_last)])

    logger.info("=" * 70)
    logger.info("TRACKING COMPLETED... DELETING FRAMES")
    for f in [pth_chunk_temp, pth_frames]:
        if f.is_dir():
            for pth_file in f.glob("*.jpg"):
                pth_file.unlink()
            f.rmdir()


def deframe_video(str_video: str) -> Path:
    pth_video: Path = utils.Directories.pth_videos / f"{str_video}.mp4"
    if not pth_video.exists():
        raise FileNotFoundError(f"Video file not found: {pth_video}")

    pth_frames: Path = utils.Directories.pth_data / "frames_temp" / str_video
    pth_frames.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(pth_video))
    int_frame_idx: int = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pth_frame: Path = pth_frames / f"{int_frame_idx:06d}.jpg"
        cv2.imwrite(str(pth_frame), frame)
        int_frame_idx += 1
    cap.release()

    return pth_frames


if __name__ == "__main__":
    raise SystemExit(main())