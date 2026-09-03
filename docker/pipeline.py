"""
End-to-end deployment pipeline: video -> (A) tracked segmentation (CSV + overlay video) -> (B) kinematics -> OSATS scores.
(A) Runs the Mask2Former ONNX model every --int_fps frames to anchor the active instrument class and polygon.
    These sparse anchor predictions are temporally smoothed (moving mode filter, see temporal_smoothing()) 
    before the mask is propagated across the remaining frames with the SAM 2 ONNX encoder/decoder (point + previous-mask prompting). 
    Results are saved as both a csv, and an annotated overlay video (saved in --outputs).
(B) Feeds the resulting kinematic summary into the 6 trained OSATS SVR regressors to produce a final predicted score.

To execute this script, run:
    python scripts/pipeline.py --str_video <str> (defaults to the first available .mp4 file in the input directory)
    [--pth_inputs <path>] [--pth_outputs <path>]
    [--int_fps <int>] [--int_clip <int>] [--int_size <int>] 
    [--pth_mask2former_onnx <path>] [--pth_sam2_encoder <path>] [--pth_sam2_decoder <path>] [--pth_models_osats <path>]
Example:
    python scripts/pipeline.py --str_video Platinum
"""
import argparse
import csv
import cv2
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import savgol_filter
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full segmentation-tracking + OSATS pipeline on a video.")

    parser.add_argument("--str_video", type=str, default=None,
                        help="Video name (without extension) found under the input directory," \
                        "otherwise look for the first available .mp4 file.")
    parser.add_argument("--pth_inputs", type=Path, default="inputs",
                        help="Path to the input video.")
    parser.add_argument("--pth_outputs", type=Path, default="outputs",
                        help="Path to the output tracked video, csv file, and other results.")

    parser.add_argument("--int_fps", type=int, default=25,
                        help="Video framerate, used for kinematics, the output video, and tracking anchor interval (default: 25).")
    parser.add_argument("--int_clip", type=int, default=7,
                        help="Temporal smoothing window (in anchor frames) applied to Mask2Former predictions before SAM 2 tracking (default: 7).")
    parser.add_argument("--int_size", type=int, default=720,
                        help="Canonical square frame size, matching data/frames/ convention (default: 720).")

    parser.add_argument("--pth_mask2former_onnx", type=Path, default="models/mask2former.onnx",
                        help="Path to the Mask2Former ONNX model (mask2former + dinov2 backbone).")
    parser.add_argument("--pth_sam2_encoder", type=Path, default="models/sam2_encoder.onnx",
                        help="Path to the SAM 2 encoder ONNX model.")
    parser.add_argument("--pth_sam2_decoder", type=Path, default="models/sam2_decoder.onnx",
                        help="Path to the SAM 2 decoder ONNX model.")
    parser.add_argument("--pth_models_osats", type=Path, default="models",
                        help="Path to the directory containing the trained OSATS model files.")

    return parser


def main(argv: list[str] | None = None) -> None:
    logger.info("=" * 70)
    logger.info("LOADING PARAMETERS...")
    args = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    str_video: str = str(args.str_video)
    pth_inputs: Path = Path(args.pth_inputs)
    pth_outputs: Path = Path(args.pth_outputs)
    int_fps: int = int(args.int_fps)
    int_clip: int = int(args.int_clip)
    int_size: int = int(args.int_size)
    pth_mask2former_onnx: Path = Path(args.pth_mask2former_onnx)
    pth_sam2_encoder: Path = Path(args.pth_sam2_encoder)
    pth_sam2_decoder: Path = Path(args.pth_sam2_decoder)
    pth_models_osats: Path = Path(args.pth_models_osats)

    pth_video: Path = pth_inputs / f"{str_video}.mp4"
    if not pth_video.is_file():
        lst_pth_videos: list[Path] = sorted(pth_inputs.glob("*.mp4"))
        if not lst_pth_videos:
            raise FileNotFoundError(f"Video file not found: {pth_video}. No mp4 files found in the input directory.")
        logger.warning(f"Video file not found: {pth_video}. Using the first available mp4 file: {lst_pth_videos[0]}")
        pth_video = lst_pth_videos[0]
    str_video = pth_video.stem

    pth_outputs.mkdir(parents=True, exist_ok=True)
    pth_predictions_csv: Path = pth_outputs / f"{str_video}_predictions.csv"
    pth_kinematics_csv: Path = pth_outputs / f"{str_video}_kinematics.csv"
    pth_segmentation_csv: Path = pth_outputs / f"{str_video}_segmentation.csv"
    pth_video_out: Path = pth_outputs / f"{str_video}.mp4"

    logger.info("=" * 70)
    logger.info("LOADING CLASSIFICATION, SEGMENTATION, AND TRACKING MODELS...")
    get_device() # asserts CUDA is available (required by the Mask2Former/SAM2 models)
    arr_weight_map: np.ndarray = Kinematics.create_mountain_gradient(int_size=int_size)
    mask2former = Mask2FormerModel(pth_onnx=pth_mask2former_onnx, int_input_size=518, int_output_size=int_size)
    sam2 = SAM2Tracker(pth_encoder=pth_sam2_encoder, pth_decoder=pth_sam2_decoder, int_output_size=int_size)

    logger.info("=" * 70)
    logger.info("LOADING VIDEO...")
    cap = cv2.VideoCapture(str(pth_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {pth_video}!")

    logger.info("=" * 70)
    logger.info("RUNNING SEGMENTATION...")
    int_total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lst_anchor_frames: list[int] = list(range(0, int_total_frames, int_fps))
    lst_anchor_classes: list[int] = []
    lst_anchor_polygons: list[list[tuple[int, int]]] = []
    for int_anchor_frame in tqdm(lst_anchor_frames, desc="Mask2Former anchors"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int_anchor_frame)
        ret, frame = cap.read()
        if not ret:
            lst_anchor_classes.append(0)
            lst_anchor_polygons.append([])
            continue
        arr_frame_720: np.ndarray = cv2.resize(frame, (int_size, int_size), interpolation=cv2.INTER_LINEAR)
        int_class, lst_polygon = mask2former.predict(arr_frame=arr_frame_720)
        lst_anchor_classes.append(int_class)
        lst_anchor_polygons.append(lst_polygon)
    lst_anchor_classes, lst_anchor_polygons = mask2former.temporal_smoothing(lst_clf_preds=lst_anchor_classes, lst_seg_preds=lst_anchor_polygons, int_clip=int_clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    logger.info("=" * 70)
    logger.info("RUNNING TRACKING...")
    video_renderer = VideoRenderer(pth_video_out=pth_video_out, int_fps=int_fps, int_size=int_size)

    int_active_class: int = 0
    tpl_point: tuple[float, float] | None = None
    arr_mask_input: np.ndarray | None = None
    int_anchor_idx: int = 0

    with open(pth_segmentation_csv, mode="w", newline="", encoding="utf-8") as f_csv, tqdm(total=int_total_frames, desc="SAM2 tracking") as pbar:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["INT_FRAME", "STR_CLASS_PRED", "LST_SEGMENTATION_PRED"])

        int_frame_idx: int = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            arr_frame_720: np.ndarray = cv2.resize(frame, (int_size, int_size), interpolation=cv2.INTER_LINEAR)

            if int_frame_idx % int_fps == 0:
                int_active_class: int = lst_anchor_classes[int_anchor_idx]
                lst_polygon: list[tuple[int, int]] = lst_anchor_polygons[int_anchor_idx]
                int_anchor_idx += 1
                if int_active_class != 0 and len(lst_polygon) >= 3:
                    tpl_point = SAM2Tracker.polygon_centroid(lst_polygon)
                    arr_mask_input = sam2.polygon_to_mask_input(lst_polygon, int_size)
                else:
                    int_active_class, lst_polygon, tpl_point, arr_mask_input = 0, [], None, None
            elif int_active_class == 0:
                lst_polygon = []
            else:
                lst_polygon, arr_mask_input, tpl_point = sam2.track(arr_frame_720, tpl_point, arr_mask_input) # type: ignore
                if len(lst_polygon) < 3:
                    int_active_class, lst_polygon, tpl_point, arr_mask_input = 0, [], None, None

            str_class: str = Instruments.dct_instruments_reverse[int_active_class]
            str_polygon: str = ";".join(str(coord) for pt in lst_polygon for coord in pt) if lst_polygon else "0;0"
            csv_writer.writerow([int_frame_idx, str_class, str_polygon])
            video_renderer.write(arr_frame_720, lst_polygon, int_active_class, str_class)

            int_frame_idx += 1
            pbar.update(1)

    cap.release()
    video_renderer.release()
    logger.info(f"Saved tracking CSV to: {pth_segmentation_csv}")
    logger.info(f"Saved overlay video to: {pth_video_out}")

    logger.info("=" * 70)
    logger.info("CALCULATING KINEMATICS...")
    df_stats: pd.DataFrame = Kinematics.compute_kinematics_for_video(pth_tracks=pth_segmentation_csv, int_size=int_size, int_fps=int_fps, arr_weight_map=arr_weight_map)
    df_stats.to_csv(pth_kinematics_csv, index=False)
    logger.info(f"Saved kinematics to: {pth_kinematics_csv}")

    logger.info("=" * 70)
    logger.info("PREDICTING OSATS...")
    osats_predictor = OsatsPredictor(pth_models_osats=pth_models_osats)
    dct_scores: dict[str, float] = osats_predictor.predict(df_stats)
    pd.DataFrame([dct_scores]).to_csv(pth_predictions_csv, index=False)
    for str_attribute, flt_score in dct_scores.items():
        logger.info(f"  {str_attribute}: {flt_score:.2f}")
    logger.info(f"Saved OSATS predictions to {pth_predictions_csv}")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED!!!")


def get_device() -> torch.device:
    """
    Returns the appropriate torch device for computations.
    Args:
        None
    Returns:
        torch.device: The device to be used for computations.
    Raises:
        RuntimeError: If CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32 matrix multiplication precision to high for Tensor Core acceleration")
    else:
        raise RuntimeError("CUDA is not available so script will not run. Please check your GPU setup.")
    return torch.device("cuda")


class Instruments:
    dct_instruments: dict[str, int] = {
        "no_instrument": 0,
        "cup_forceps": 1,
        "disc_dissector": 2,
        "dural_scissors": 3,
        "irrigation_syringe": 4,
        "kerrisons": 5,
        "nasal_cutting_forceps": 6,
        "pituitary_rongeurs": 7,
        "retractable_knife": 8,
        "ring_curette": 9,
        "spatula_dissector": 10,
        "suction": 11,
        "surgical_drill": 12,
    }
    dct_instruments_reverse: dict[int, str] = {v: k for k, v in dct_instruments.items()}


class Mask2FormerModel:
    """Wraps the Mask2Former ONNX model: preprocesses a frame and returns the dominant (class, polygon)."""
    def __init__(self, pth_onnx: Path, int_input_size: int = 518, int_output_size: int = 720):
        """
        Initializes the Mask2Former model.
        Args:
            pth_onnx (Path): Path to the Mask2Former ONNX model.
            int_input_size (int, optional): The input size for the model. Defaults to 518.
            int_output_size (int, optional): The output size for the scaled polygon. Defaults to 720.
        Raises:
            FileNotFoundError: If the ONNX model file is not found.
        """
        if not pth_onnx.is_file():
            raise FileNotFoundError(f"Mask2Former ONNX model not found: {pth_onnx}")
        self.int_input_size: int = int_input_size
        self.int_output_size: int = int_output_size
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess: ort.InferenceSession = ort.InferenceSession(str(pth_onnx), providers=providers)
        self.str_input_name: str = self.sess.get_inputs()[0].name

    def predict(self, arr_frame: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        """
        Runs the model on a frame and returns the dominant non-zero class and its largest contour (scaled to int_target_size).
        Args:
            arr_frame (np.ndarray): The input frame in BGR format.
        Returns:
            tuple[int, list[tuple[int, int]]]: The dominant class and its largest contour polygon.
        """
        arr_img: np.ndarray = cv2.resize(arr_frame, (self.int_input_size, self.int_input_size), interpolation=cv2.INTER_LINEAR)
        arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr_img = arr_img.transpose(2, 0, 1)[None]
        arr_input: np.ndarray = np.ascontiguousarray(arr_img, dtype=np.float32)

        arr_class_map: np.ndarray = self.sess.run(None, {self.str_input_name: arr_input})[0].squeeze().astype(np.int32) # type: ignore
        arr_nonzero: np.ndarray = arr_class_map[arr_class_map != 0]
        if arr_nonzero.size == 0:
            return 0, []
        int_class: int = int(np.bincount(arr_nonzero).argmax())

        arr_binary: np.ndarray = (arr_class_map == int_class).astype(np.uint8)
        contours, _ = cv2.findContours(arr_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, []
        arr_largest: np.ndarray = max(contours, key=cv2.contourArea)
        flt_scale: float = self.int_output_size / arr_class_map.shape[0]
        lst_polygon: list[tuple[int, int]] = [(int(flt_pt[0][0] * flt_scale), int(flt_pt[0][1] * flt_scale)) for flt_pt in arr_largest] # type: ignore
        if len(lst_polygon) < 3:
            return 0, []
        return int_class, lst_polygon

    @staticmethod
    def temporal_smoothing(lst_clf_preds: list[int], lst_seg_preds: list[list[tuple[int, int]]], int_clip: int = 5, bl_online: bool = False) -> tuple[list[int], list[list[tuple[int, int]]]]:
        """
        Smooths class predictions using a moving mode filter over a sliding window.
        Args:
            lst_clf_preds: List of predicted class indices.
            lst_seg_preds: List of predicted segmentations corresponding to each frame.
            int_clip: The number of frames to aggregate over (window size).
            bl_online: The smoothing mode to use.
                True: Aggregates from (current - int_clip + 1) up to the current frame.
                False: Aggregates int_clip // 2 frames before and after the current frame (this is the default).
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
                    x != 0, # if still tied, use the non-zero classes
                    len(lst_window) - 1 - lst_window[::-1].index(x), # if still tied, prefer the most recent occurrence if tied
                )
            )
            lst_smoothed_clf_preds.append(int_mode)

        for i in range(len(lst_smoothed_clf_preds)):
            if lst_smoothed_clf_preds[i] == 0:
                lst_seg_preds[i] = [(0, 0)]

        return lst_smoothed_clf_preds, lst_seg_preds


class SAM2Tracker:
    """Wraps the SAM 2 encoder/decoder ONNX sessions to propagate a mask forward one frame at a time."""
    def __init__(self, pth_encoder: Path, pth_decoder: Path, int_encoder_size: int = 1024, int_output_size: int = 720):
        """
        Initializes the SAM2Tracker with the given encoder and decoder ONNX model paths and encoder/output sizes.
        Args:
            pth_encoder (Path): Path to the SAM 2 encoder ONNX model.
            pth_decoder (Path): Path to the SAM 2 decoder ONNX model.
            int_encoder_size (int, optional): Size of the encoder input (default: 1024, SAM2 encoder input size).
            int_output_size (int, optional): Size of the output mask (default: 720).
        """
        if not pth_encoder.is_file() or not pth_decoder.is_file():
            raise FileNotFoundError(f"SAM 2 ONNX model(s) not found: {pth_encoder}, {pth_decoder}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.encoder_ort: ort.InferenceSession = ort.InferenceSession(str(pth_encoder), providers=providers)
        self.decoder_ort: ort.InferenceSession = ort.InferenceSession(str(pth_decoder), providers=providers)
        self.int_encoder_size: int = int_encoder_size
        self.int_output_size: int = int_output_size

    def track(self, arr_frame: np.ndarray, tpl_point: tuple[float, float], arr_mask_input: np.ndarray | None) -> tuple[list[tuple[int, int]], np.ndarray | None, tuple[float, float] | None]:
        """
        Segments the prompted object in arr_frame and returns (polygon at int_output_size, next mask_input, next point) for the next call, 
        or (empty polygon, None, None) if the mask degenerates.
        Args:
            arr_frame (np.ndarray): The current video frame.
            arr_mask_input (np.ndarray | None): The previous mask input, if available.
        Returns:
            tuple[list[tuple[int, int]], np.ndarray | None, tuple[float, float] | None]: A tuple containing the polygon coordinates, the next mask input, and the next point prompt.
        """
        arr_img: np.ndarray = cv2.resize(arr_frame, (self.int_encoder_size, self.int_encoder_size), interpolation=cv2.INTER_LINEAR)
        arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr_img = (arr_img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr_img = arr_img.transpose(2, 0, 1)[None]
        arr_input: np.ndarray = np.ascontiguousarray(arr_img, dtype=np.float32)
    
        lst_outputs = self.encoder_ort.run(None, {"image": arr_input})
        dct_feats: dict[str, np.ndarray] = dict(zip([o.name for o in self.encoder_ort.get_outputs()], lst_outputs)) # type: ignore

        flt_scale: float = self.int_encoder_size / self.int_output_size
        arr_point_coords: np.ndarray = np.array([[[tpl_point[0] * flt_scale, tpl_point[1] * flt_scale]]], dtype=np.float32)
        arr_point_labels: np.ndarray = np.ones((1, 1), dtype=np.float32)
        if arr_mask_input is None:
            arr_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            arr_has_mask_input: np.ndarray = np.zeros((1,), dtype=np.float32)
        else:
            arr_has_mask_input = np.ones((1,), dtype=np.float32)

        arr_masks, arr_iou_preds = self.decoder_ort.run(None, {
            "image_embed": dct_feats["image_embed"],
            "high_res_feats_0": dct_feats["high_res_feats_0"],
            "high_res_feats_1": dct_feats["high_res_feats_1"],
            "point_coords": arr_point_coords,
            "point_labels": arr_point_labels,
            "mask_input": arr_mask_input,
            "has_mask_input": arr_has_mask_input,
        })

        int_best: int = int(np.argmax(arr_iou_preds[0])) # type: ignore
        arr_best_mask_256: np.ndarray = arr_masks[0, int_best] # type: ignore

        arr_mask_resized: np.ndarray = cv2.resize(arr_best_mask_256, (self.int_output_size, self.int_output_size), interpolation=cv2.INTER_LINEAR)
        arr_binary: np.ndarray = (arr_mask_resized > 0.0).astype(np.uint8)
        if arr_binary.sum() == 0:
            return [], None, None

        contours, _ = cv2.findContours(arr_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], None, None
        arr_largest: np.ndarray = max(contours, key=cv2.contourArea)
        lst_polygon: list[tuple[int, int]] = [(int(flt_pt[0][0]), int(flt_pt[0][1])) for flt_pt in arr_largest] # type: ignore
        if len(lst_polygon) < 3:
            return [], None, None

        tpl_next_point: tuple[float, float] = self.polygon_centroid(lst_polygon)
        arr_next_mask_input: np.ndarray = arr_best_mask_256[None, None].astype(np.float32)
        return lst_polygon, arr_next_mask_input, tpl_next_point

    @staticmethod
    def polygon_centroid(lst_polygon: list[tuple[int, int]]) -> tuple[float, float]:
        """Computes the (unweighted) centroid of a polygon, used as the next frame's SAM 2 point prompt."""
        arr_pts: np.ndarray = np.array(lst_polygon, dtype=np.int32)
        dct_moments = cv2.moments(arr_pts)
        if dct_moments["m00"] != 0:
            return dct_moments["m10"] / dct_moments["m00"], dct_moments["m01"] / dct_moments["m00"]
        return float(np.mean(arr_pts[:, 0])), float(np.mean(arr_pts[:, 1]))

    @staticmethod
    def polygon_to_mask_input(lst_polygon: list[tuple[int, int]], int_size: int) -> np.ndarray:
        """
        Rasterizes a polygon and converts it to pseudo-logits usable as the SAM 2 decoder's mask_input.
        Args:
            lst_polygon: List of (x, y) coordinates defining a polygon.
            int_size: Size of the frame.
        Returns:
            np.ndarray: Pseudo-logits mask suitable for SAM 2 decoder's mask_input.
        """
        arr_mask: np.ndarray = np.zeros((int_size, int_size), dtype=np.uint8)
        cv2.fillPoly(arr_mask, [np.array(lst_polygon, dtype=np.int32)], 1)
        arr_mask_256: np.ndarray = cv2.resize(arr_mask, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        arr_logits: np.ndarray = (arr_mask_256 * 2.0 - 1.0) * 10.0 # +10 inside / -10 outside
        return arr_logits[None, None].astype(np.float32)


class VideoRenderer:
    """Draws polygon and classification overlays onto frames and writes them to the output overlay video."""
    def __init__(self, pth_video_out: Path, int_fps: int, int_size: int):
        """
        Initialize the VideoRenderer with the output video path, FPS, and frame size.
        Args:
            pth_video_out: Path to the output video file.
            int_fps: Frames per second for the output video.
            int_size: Size of the video frames (assumed square).
        """
        self.video_writer = cv2.VideoWriter(str(pth_video_out), cv2.VideoWriter_fourcc(*"mp4v"), int_fps, (int_size, int_size)) # type: ignore

    def draw_overlay(self, arr_frame: np.ndarray, lst_polygon: list[tuple[int, int]], int_class: int, str_class: str) -> np.ndarray:
        arr_frame_out: np.ndarray = arr_frame.copy()
        arr_colour_rng: np.random.Generator = np.random.default_rng(int_class * 97 + 13) # deterministic color based on class
        arr_color: np.ndarray = arr_colour_rng.integers(64, 255, size=3)
        tpl_color: tuple[int, int, int] = int(arr_color[0]), int(arr_color[1]), int(arr_color[2])
        if len(lst_polygon) >= 3:
            arr_pts: np.ndarray = np.array(lst_polygon, dtype=np.int32)
            arr_frame_overlay: np.ndarray = arr_frame_out.copy()
            cv2.fillPoly(arr_frame_overlay, [arr_pts], tpl_color)
            arr_frame_out = cv2.addWeighted(arr_frame_overlay, 0.35, arr_frame_out, 0.65, 0)
            cv2.polylines(arr_frame_out, [arr_pts], isClosed=True, color=tpl_color, thickness=2)

        # classification label always shown top-right, regardless of whether a polygon was found
        (int_text_w, int_text_h), _ = cv2.getTextSize(str_class, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tpl_text_pos: tuple[int, int] = (arr_frame_out.shape[1] - int_text_w - 15, int_text_h + 15)
        cv2.putText(arr_frame_out, str_class, tpl_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, tpl_color, 2, cv2.LINE_AA)
        return arr_frame_out

    def write(self, arr_frame: np.ndarray, lst_polygon: list[tuple[int, int]], int_class: int, str_class: str) -> None:
        self.video_writer.write(self.draw_overlay(arr_frame, lst_polygon, int_class, str_class))

    def release(self) -> None:
        self.video_writer.release()


class Kinematics:
    """Wraps calculate_kinematics.py's per-block helpers to summarize a video's tracking CSV."""
    @staticmethod
    def get_kinematics_attributes() -> list[str]:
        return [
            "FLT_TIME", # Time duration of all instruments in a video
            "FLT_DISTANCE", # Total distance traveled by all instruments
            "FLT_SPEED", # Average speed across all instruments
            "FLT_ACCELERATION", # Average acceleration across all instruments
        ]

    @staticmethod
    def compute_kinematics_for_video(pth_tracks: Path, int_size: int, int_fps: int, arr_weight_map: np.ndarray) -> pd.DataFrame:
        df_tracks: pd.DataFrame = pd.read_csv(pth_tracks)
        df_blocks: pd.Series = df_tracks["STR_CLASS_PRED"].ne(df_tracks["STR_CLASS_PRED"].shift()).cumsum()
        df_grouped_blocks: pd.Series = df_tracks.groupby(["STR_CLASS_PRED", df_blocks])["INT_FRAME"].agg(list)
        dct_class_frames: dict[str, list[list[int]]] = df_grouped_blocks.groupby(level=0).agg(list).to_dict() # type: ignore

        df_stats: pd.DataFrame = pd.DataFrame(
            np.zeros((len(Instruments.dct_instruments), 4), dtype=float),
            index=range(len(Instruments.dct_instruments)),
            columns=Kinematics.get_kinematics_attributes(),
        )
        for str_class in Instruments.dct_instruments.keys():
            int_class: int = Instruments.dct_instruments[str_class]
            if str_class not in dct_class_frames.keys() or int_class <= 0:
                continue
            for lst_frames_block in dct_class_frames[str_class]:
                df_tracks_chunk: pd.DataFrame = df_tracks[df_tracks["INT_FRAME"].isin(lst_frames_block)].copy()
                for row in df_tracks_chunk.itertuples(index=True):
                    lst_coords: list[float] = [float(coord) for coord in str(row.LST_SEGMENTATION_PRED).split(";")]
                    lst_polygon: list[tuple[int, int]] = [(int(lst_coords[i]), int(lst_coords[i + 1])) for i in range(0, len(lst_coords), 2)]
                    flt_centroid_x, flt_centroid_y = Kinematics.calculate_centroid(lst_coords=lst_polygon, arr_weight_map=arr_weight_map, int_size=int_size)
                    df_tracks_chunk.loc[row.Index, "FLT_CENTROID_X"] = flt_centroid_x
                    df_tracks_chunk.loc[row.Index, "FLT_CENTROID_Y"] = flt_centroid_y
                df_stats.loc[int_class] += Kinematics.calculate_kinematics(df_tracks_chunk=df_tracks_chunk, int_fps=int_fps)
        df_stats = df_stats / (len(df_tracks) - 1)

        df_stats.loc[0] = df_stats.sum(axis=0)
        df_stats = df_stats.sort_index()
        df_stats.insert(0, "INT_CLASS", df_stats.index.astype(int))
        return df_stats

    @staticmethod
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

    @staticmethod
    def calculate_centroid(lst_coords: list[tuple[int, int]], arr_weight_map: np.ndarray, int_size: int = 720) -> tuple[float, float]:
        """
        Calculate the center of mass and area from a list of polygon coordinates,
        biasing the centroid heavily toward the center of the image.
        Args:
            lst_coords: List of (x, y) coordinates defining a polygon, or None.
            arr_weight_map: Pre-computed 2D numpy array of spatial weights.
            int_size: Size of a frame (default: 720).
        Returns:
            tuple[float, float]: Weighted Centroid coordinates (cX, cY).
        """
        if not lst_coords or len(lst_coords) < 3:
            return int_size / 2.0, int_size / 2.0

        arr_coords: np.ndarray = np.array(lst_coords, dtype=np.int32)
        arr_mask: np.ndarray = np.zeros((int_size, int_size), dtype=np.float32)
        cv2.fillPoly(arr_mask, [arr_coords], 1.0)
        arr_weighted_mask: np.ndarray = arr_mask * arr_weight_map
        dct_moments = cv2.moments(arr_weighted_mask, binaryImage=False)
        if dct_moments["m00"] != 0:
            flt_x = dct_moments["m10"] / dct_moments["m00"]
            flt_y = dct_moments["m01"] / dct_moments["m00"]
            return flt_x, flt_y
        else: 
            flt_x = float(np.mean(arr_coords[:, 0]))
            flt_y = float(np.mean(arr_coords[:, 1]))
            return flt_x, flt_y

    @staticmethod
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


class OsatsPredictor:
    """Runs the trained OSATS SVR regressors on a video's total kinematic summary."""
    def __init__(self, pth_models_osats: Path):
        """
        Initialize the OSATS predictor with pre-trained models.
        Args:
            pth_models_osats: Path to the directory containing the trained OSATS model files.
        """
        lst_osats_models: list[str] = [
            "RESPECT_FOR_TISSUE", 
            "TIME_AND_MOTION",
            "INSTRUMENT_HANDLING",
            "FLOW_OF_OPERATION",
            "KNOWLEDGE_OF_INSTRUMENTS",
            "KNOWLEDGE_OF_PROCEDURE",
        ]
        self.dct_models = {str_attribute: joblib.load(pth_models_osats / f"{str_attribute}.joblib") for str_attribute in lst_osats_models}
        self.lst_kinematics_attributes: list[str] = Kinematics.get_kinematics_attributes()

    def predict(self, df_stats: pd.DataFrame) -> dict[str, float]:
        df_totals: pd.Series = df_stats[df_stats["INT_CLASS"] == 0].iloc[0]
        arr_features: np.ndarray = np.array([[df_totals[str_col] for str_col in self.lst_kinematics_attributes]])

        dct_scores: dict[str, float] = {}
        flt_total: float = 0.0
        for str_attribute, model in self.dct_models.items():
            flt_pred: float = float(np.clip(model.predict(arr_features)[0], 1, 5))
            dct_scores[str_attribute] = flt_pred
            flt_total += flt_pred
        dct_scores["INT_TOTAL"] = flt_total
        return dct_scores


if __name__ == "__main__":
    raise SystemExit(main())
