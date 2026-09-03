"""
Extract frames from a video that match the INT_FRAME values in an annotations CSV.

Usage:
    python scripts/extract_annotated_frames.py --str_video Barium
"""
import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd

import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract annotated frames from a video.")
    parser.add_argument("--str_video", required=True, help="Video name (without extension), e.g. Barium.")
    parser.add_argument("--pth_video", type=Path, default=None,
                        help="Path to video file (default: data/videos/<str_video>.mp4).")
    parser.add_argument("--pth_annotations", type=Path, default=None,
                        help="Path to annotations CSV (default: data/annotations/<str_video>.csv).")
    parser.add_argument("--pth_output_dir", type=Path, default=None,
                        help="Output directory for frames (default: data/frames/<str_video>).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    pth_video: Path = args.pth_video or (utils.Directories.pth_videos / f"{args.str_video}.mp4")
    pth_annotations: Path = args.pth_annotations or (utils.Directories.pth_annotations / f"{args.str_video}.csv")
    pth_output_dir: Path = args.pth_output_dir or (utils.Directories.pth_frames / args.str_video)

    if not pth_video.is_file():
        raise FileNotFoundError(f"Video not found: {pth_video}")
    if not pth_annotations.is_file():
        raise FileNotFoundError(f"Annotations not found: {pth_annotations}")

    pth_output_dir.mkdir(parents=True, exist_ok=True)

    df_annotations: pd.DataFrame = pd.read_csv(pth_annotations)
    set_target_frames: set[int] = set(df_annotations["INT_FRAME"].astype(int).tolist())
    logger.info(f"Found {len(set_target_frames)} unique target frames in {pth_annotations}")

    cap = cv2.VideoCapture(str(pth_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {pth_video}")

    int_total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video has {int_total_frames} total frames")

    int_max_target: int = max(set_target_frames)
    if int_max_target >= int_total_frames:
        logger.warning(f"Max target frame {int_max_target} >= total video frames {int_total_frames}. "
                        f"Some frames will be unavailable.")

    int_extracted: int = 0
    int_frame_idx: int = 0
    int_missing: int = 0

    while True:
        bl_ret, npy_frame = cap.read()
        if not bl_ret:
            break

        if int_frame_idx in set_target_frames:
            pth_out: Path = pth_output_dir / f"{int_frame_idx:06d}.png"
            cv2.imwrite(str(pth_out), npy_frame)
            int_extracted += 1

        int_frame_idx += 1

        if int_extracted == len(set_target_frames):
            logger.info("All target frames extracted, stopping early.")
            break

    cap.release()

    int_missing = len(set_target_frames) - int_extracted
    logger.info(f"Extracted {int_extracted}/{len(set_target_frames)} frames to {pth_output_dir}")
    if int_missing > 0:
        set_extracted_frames = {int(p.stem) for p in pth_output_dir.glob("*.png")}
        set_missing_frames = set_target_frames - set_extracted_frames
        logger.warning(f"Missing {int_missing} frames: {sorted(set_missing_frames)[:20]}"
                        f"{'...' if int_missing > 20 else ''}")


if __name__ == "__main__":
    raise SystemExit(main())