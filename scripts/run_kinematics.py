"""
Calculate OSATS kinematic summaries from tracked polygons.
Extracts centroid trajectories from polygon masks, applies Savitzky-Golay smoothing, and computes kinematic metrics. 

To execute this script, run:
    python scripts/run_kinematics.py --int_model <model_number> 
    [--str_split <video_split>] [--int_size <frame_size>] [--int_fps <fps>]
Example:
    --int_model 1
    --int_model 2 --str_split ss_val_test --int_size 720 --int_fps 25
"""
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

import utils

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate OSATS Kinematic Summaries from Tracked Polygons.")

    parser.add_argument("--int_model", type=str, required=True, 
                        help="Model number for saved annotations found in results.")
    parser.add_argument("--str_split", type=str, default="ss_val_test", choices=utils.Videos.dct_split_videos.keys(),
                        help="Video split to process (default: 'ss_val_test').")
    
    parser.add_argument("--int_size", type=int, default=720,
                        help="Size of a frame (default: 720).")
    parser.add_argument("--int_fps", type=int, default=25, 
                        help="Video framerate (default: 25)")
    
    return parser


def main(argv: list[str] | None = None) -> None:
    logger.info("=" * 70)
    logger.info("LOADING PARAMETERS...")
    args = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    pth_tracking = utils.Directories.pth_tracking / args.int_model
    pth_kinematics = utils.Directories.pth_kinematics / args.int_model
    pth_kinematics.mkdir(parents=True, exist_ok=True)
    if not pth_tracking.exists():
        raise FileNotFoundError(f"Tracking path '{pth_tracking}' not found.")

    arr_weight_map: np.ndarray = create_mountain_gradient(int_size=args.int_size)

    for str_video in utils.Videos.dct_split_videos[args.str_split]:
        logger.info("=" * 70)
        logger.info(f"CALCULATING KINEMATICS FOR VIDEO {str_video}")
        pth_tracks: Path = pth_tracking / f"{str_video}.csv"
        if not pth_tracks.exists():
            raise FileNotFoundError(f"Coordinates file '{pth_tracks}' not found.")
        df_tracks: pd.DataFrame = pd.read_csv(pth_tracks)
        df_blocks: pd.Series = df_tracks["STR_CLASS_PRED"].ne(df_tracks["STR_CLASS_PRED"].shift()).cumsum()
        df_grouped_blocks: pd.Series = df_tracks.groupby(["STR_CLASS_PRED", df_blocks])["INT_FRAME"].agg(list)
        dct_class_frames: dict[str, list[list[int]]] = df_grouped_blocks.groupby(level=0).agg(list).to_dict() # type: ignore

        df_stats: pd.DataFrame = pd.DataFrame(
            np.zeros((utils.Instruments.int_classes, 4), dtype=float),
            index=range(utils.Instruments.int_classes),
            columns=[col for col in utils.Osats.lst_kinematics_attributes if col != "INT_INSTRUMENTS"], 
        )
        for str_class in utils.Instruments.dct_str_int_class.keys():
            int_class: int = utils.Instruments.dct_str_int_class[str_class]
            if str_class not in dct_class_frames.keys() or int_class <= 0:
                continue
            for lst_frames_block in dct_class_frames[str_class]:
                df_tracks_chunk: pd.DataFrame = df_tracks[df_tracks["INT_FRAME"].isin(lst_frames_block)].copy()
                for row in df_tracks_chunk.itertuples(index=True):
                    lst_coords: list[float] = [float(coord) for coord in str(row.LST_SEGMENTATION_PRED).split(";")]
                    lst_polygon: list[tuple[int, int]] = [(int(lst_coords[i]), int(lst_coords[i + 1])) for i in range(0, len(lst_coords), 2)]
                    flt_centroid_x, flt_centroid_y, flt_area = calculate_centroid_and_area(lst_coords=lst_polygon, arr_weight_map=arr_weight_map, int_size=args.int_size)
                    df_tracks_chunk.loc[row.Index, "FLT_CENTROID_X"] = flt_centroid_x
                    df_tracks_chunk.loc[row.Index, "FLT_CENTROID_Y"] = flt_centroid_y
                    df_tracks_chunk.loc[row.Index, "FLT_AREA"] = flt_area
                df_stats.loc[int_class] += calculate_kinematics(df_tracks_chunk=df_tracks_chunk, int_fps=args.int_fps)
        df_stats = df_stats / (len(df_tracks) - 1)

        df_stats.loc[0] = df_stats.sum(axis=0)
        df_stats = df_stats.sort_index()
        df_stats.insert(0, "INT_CLASS", df_stats.index.astype(int))

        df_stats.to_csv(pth_kinematics / f"{str_video}.csv", index=False)

    logger.info("=" * 70)
    logger.info("ALL KINEMATICS SAVED!!")


def calculate_centroid_and_area(lst_coords: list[tuple[int, int]], arr_weight_map: np.ndarray, int_size: int = 720) -> tuple[float, float, float]:
    """
    Calculate the center of mass and area from a list of polygon coordinates,
    biasing the centroid heavily toward the center of the image.
    Args:
        lst_coords: List of (x, y) coordinates defining a polygon, or None.
        arr_weight_map: Pre-computed 2D numpy array of spatial weights.
        int_size: Size of a frame (default: 720).
    Returns:
        tuple[float, float, float]: Weighted Centroid coordinates (cX, cY) and true area.
    """
    if not lst_coords or len(lst_coords) < 3:
        return int_size / 2.0, int_size / 2.0, 0.0

    arr_coords: np.ndarray = np.array(lst_coords, dtype=np.int32)
    arr_mask: np.ndarray = np.zeros((int_size, int_size), dtype=np.float32)
    cv2.fillPoly(arr_mask, [arr_coords], 1.0)
    arr_weighted_mask: np.ndarray = arr_mask * arr_weight_map
    dct_moments = cv2.moments(arr_weighted_mask, binaryImage=False)
    if dct_moments["m00"] != 0:
        flt_area = float(cv2.contourArea(arr_coords))
        flt_x = dct_moments["m10"] / dct_moments["m00"]
        flt_y = dct_moments["m01"] / dct_moments["m00"]
        return flt_x, flt_y, flt_area
    else: 
        flt_x = float(np.mean(arr_coords[:, 0]))
        flt_y = float(np.mean(arr_coords[:, 1]))
        return flt_x, flt_y, 0.0


def create_mountain_gradient(int_size: int = 720) -> np.ndarray:
    """
    Creates a 2D spatial weight map using Chebyshev distance.
    This function generates a weight map where the center of the grid has the highest weight,
    and the weights decrease towards the edges based on the Chebyshev distance.
    For a 720x720 grid, the center weights are 360, radiating down to 1 at the borders.
    Args:
        int_size: Size of a frame (default: 720).
    Returns:
        np.ndarray: 2D array representing the spatial weight map.
    """
    arr_x: np.ndarray = np.arange(int_size)
    arr_y: np.ndarray = np.arange(int_size)
    arr_xv, arr_yv = np.meshgrid(arr_x, arr_y)
    flt_centre: float = (int_size - 1) / 2.0
    
    arr_dist_x: np.ndarray = np.abs(arr_xv - flt_centre)
    arr_dist_y: np.ndarray = np.abs(arr_yv - flt_centre)
    chebyshev_dist: np.ndarray = np.maximum(arr_dist_x, arr_dist_y)
    
    flt_weight_max: float = (int_size / 2) + 0.5
    arr_weight_map: np.ndarray = flt_weight_max - chebyshev_dist
    return arr_weight_map.astype(np.float32)


def calculate_kinematics(df_tracks_chunk: pd.DataFrame, int_fps: int = 25) -> list[float]:
    """
    Calculate kinematic metrics for one tracking block.
    Args:
        df_tracks_chunk: DataFrame containing tracking data for a single block.
        int_fps: Frames per second of the tracking data (default: 25).
    Returns:
        list[float]: Kinematic metrics [time, distance, speed, acceleration].
    """
    int_length: int = len(df_tracks_chunk)
    if int_length < 2:
        return [0.0, 0.0, 0.0, 0.0]
    if int_fps <= 0:
        raise ValueError("int_fps must be greater than 0.")

    flt_dt: float = 1.0 / int_fps

    df_tracks_chunk = df_tracks_chunk.copy()
    df_tracks_chunk["x_smooth"] = df_tracks_chunk["FLT_CENTROID_X"].astype(float)
    df_tracks_chunk["y_smooth"] = df_tracks_chunk["FLT_CENTROID_Y"].astype(float)

    int_poly_order: int = 3
    int_window_length: int = min(11, int_length)
    if int_window_length % 2 == 0:
        int_window_length -= 1

    if int_window_length > int_poly_order:
        df_tracks_chunk["x_smooth"] = savgol_filter(df_tracks_chunk["x_smooth"], int_window_length, int_poly_order)
        df_tracks_chunk["y_smooth"] = savgol_filter(df_tracks_chunk["y_smooth"], int_window_length, int_poly_order)

    flt_time: float = float(int_length - 1)

    arr_x_distance = np.diff(df_tracks_chunk["x_smooth"])
    arr_y_distance = np.diff(df_tracks_chunk["y_smooth"])
    flt_distance: float = float(np.sqrt(arr_x_distance**2 + arr_y_distance**2).sum())

    arr_v_x = np.gradient(df_tracks_chunk["x_smooth"].to_numpy(), flt_dt)
    arr_v_y = np.gradient(df_tracks_chunk["y_smooth"].to_numpy(), flt_dt)
    flt_speed: float = float(np.sqrt(arr_v_x**2 + arr_v_y**2).mean())

    arr_a_x = np.gradient(arr_v_x, flt_dt)
    arr_a_y = np.gradient(arr_v_y, flt_dt)
    flt_acceleration: float = float(np.sqrt(arr_a_x**2 + arr_a_y**2).mean())

    return [flt_time, flt_distance, flt_speed, flt_acceleration]


if __name__ == "__main__":
    raise SystemExit(main())
