"""
Hard-coded global variables to import into other scripts.
Ensures consistency and avoids hardcoding paths across multiple files.

Also includes basic utility functions.
"""
import csv
import os
import pandas as pd
import torch
from contextlib import contextmanager
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Directories:
    PARENT_DIR: Path = Path(__file__).resolve().parent.parent # path to the root of the repository

    pth_docker: Path = PARENT_DIR.joinpath("docker").resolve()

    pth_data: Path = Path(PARENT_DIR / "data").resolve()
    pth_annotations: Path = Path(pth_data / "annotations").resolve()
    pth_frames: Path = Path(pth_data / "frames").resolve()
    pth_kinematics: Path = Path(pth_data / "kinematics").resolve()
    pth_osats: Path = Path(pth_data / "osats").resolve()
    pth_results: Path = Path(pth_data / "results").resolve()
    pth_segmentation: Path = Path(pth_data / "segmentation").resolve()
    pth_tracking: Path = Path(pth_data / "tracking").resolve()
    pth_videos: Path = Path(pth_data / "videos").resolve()

    pth_models: Path = Path(PARENT_DIR / "models").resolve()
    pth_scripts: Path = Path(PARENT_DIR / "scripts").resolve()
    pth_logs: Path = Path(PARENT_DIR / "logs").resolve()

    pth_results_clf: Path = Path(pth_results / "classification.csv").resolve()
    pth_results_seg: Path = Path(pth_results / "segmentation.csv").resolve()


class Instruments:
    pth_class_metadata: Path = Path(Directories.pth_data / "class_metadata.csv").resolve()
    lst_columns: list[str] = ["STR_CLASS", "BL_INCLUDED","INT_CLASS","INT_TRAIN"]
    df_class_metadata: pd.DataFrame = pd.read_csv(pth_class_metadata)
    lst_str_classes: list[str] = sorted(df_class_metadata[df_class_metadata["BL_INCLUDED"] == 1]["STR_CLASS"].tolist())
    int_classes: int = len(lst_str_classes)
    dct_str_int_class: dict[str, int] = df_class_metadata[df_class_metadata["BL_INCLUDED"] == 1].set_index("STR_CLASS")["INT_CLASS"].astype(int).to_dict()
    dct_int_str_class: dict[int, str] = df_class_metadata[df_class_metadata["BL_INCLUDED"] == 1].set_index("INT_CLASS")["STR_CLASS"].to_dict()
    dct_volume_class: dict[str, int] = df_class_metadata[df_class_metadata["BL_INCLUDED"] == 1].set_index("STR_CLASS")["INT_TRAIN"].astype(int).to_dict()


class Videos:
    pth_video_metadata: Path = Path(Directories.pth_data / "video_metadata.csv").resolve()
    lst_columns: list[str] = [
        "STR_VIDEO", # video name as a string without the file extension (e.g., "Argon")
        "INT_START", # the starting frame number of the video as an integer (e.g., 0, 1, 2, ...)
        "INT_STOP", # the ending frame number of the video as an integer (e.g., 0, 1, 2, ...)
        "STR_SPLIT", # the split of the video as a string (e.g., "train", "val", "test")
    ]
    lst_video_metadata: list[dict[str, str]] = list(csv.DictReader(open(pth_video_metadata, 'r')))
    lst_all: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata])
    lst_train: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT"] == "train"])
    lst_ss_train1: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT"] == "ss_train1"])
    lst_ss_train2: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT"] == "ss_train2"])
    lst_val: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT"] == "val"])
    lst_test: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT"] == "test"])
    lst_ss_train: list[str] = sorted([row["STR_VIDEO"] for row in lst_video_metadata if row["STR_SPLIT_ORIGINAL"] in ["ss_train", "ss_train1", "ss_train2"]])

    dct_split_videos: dict[str, list[str]] = {
        "train": lst_train,
        "ss_train": lst_ss_train,
        "ss_train1": lst_ss_train1,
        "ss_train2": lst_ss_train2,
        "val": lst_val,
        "test": lst_test,
        "val_test": lst_val + lst_test,     
        "all": lst_all,
        "ss_val_test": lst_ss_train + lst_val + lst_test,
    }
    
    int_all: int = len(lst_all)
    int_train: int = len(lst_train)
    int_ss_train: int = len(lst_ss_train)
    int_val: int = len(lst_val)
    int_test: int = len(lst_test)


class Segmentation:
    lst_columns: list[str] = [
        "STR_VIDEO", # video name as a string without the file extension (e.g., "Argon")
        "INT_FRAME", # frame number of the video as an integer (e.g., 0, 1, 2, ...)
        "STR_CLASS", # instrument class as a string (e.g., "disc_dissector")
        "LST_SEGMENTATION", # coordinates in the form x0;y0;x1;y1;...;xn;yn (e.g., "0;0;10;10;20;20" for a triangle)
    ]
    int_columns: int = len(lst_columns)


class Osats:
    pth_osats_metadata: Path = Path(Directories.pth_data / "osats_metadata.csv").resolve()
    lst_osats_attributes: list[str] = [
        "RESPECT_FOR_TISSUE", # OSATS score for Respect for Tissue (1-5)
        "TIME_AND_MOTION", # OSATS score for Time and Motion (1-5)
        "INSTRUMENT_HANDLING", # OSATS score for Instrument Handling (1-5)
        "FLOW_OF_OPERATION", # OSATS score for Flow of Operation (1-5)
        "KNOWLEDGE_OF_INSTRUMENTS", # OSATS score for Knowledge of Instruments (1-5)
        "KNOWLEDGE_OF_PROCEDURE", # OSATS score for Knowledge of Procedure (1-5)
        "INT_TOTAL", # Total OSATS score (sum of the above scores, 6-30)
    ]
    lst_kinematics_attributes: list[str] = [
        "FLT_TIME", # Time duration of all instruments in a video
        "FLT_DISTANCE", # Total distance traveled by all instruments
        "FLT_SPEED", # Average speed across all instruments
        "FLT_ACCELERATION", # Average acceleration across all instruments
    ]
    int_columns: int = len(lst_osats_attributes)


class Models:
    lst_evaluators: list[str] = ["clf", "flex", "lstm", "cnnseg", "mmseg"]
    lst_clf_backbones: list[str] = ["convnextv2", "densenet121", "dinov2", "efficientnetv2", "moco", "resnet50", "swin"]
    lst_cnnseg_architectures: list[str] = ["unet", "unetplusplus", "deeplabv3", "deeplabv3plus"]
    lst_cnnseg_encoders: list[str] = ["convnextv2", "densenet121", "efficientnetv2", "moco", "resnet50"]
    lst_mmseg_architectures: list[str] = ["mask2former", "setrmla", "uperhead"]
    lst_mmseg_encoders: list[str] = ["dinov2"]


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


@contextmanager
def inference_context():
    """Sets up device, TF32 precision, and wraps execution in inference mode."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your GPU setup.")
    
    torch.set_float32_matmul_precision("high")
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        yield torch.device("cuda")


def is_rank_zero() -> bool:
    """
    Whether this process is the rank-0 launcher process. 
    Under Lightning's default multi-GPU strategy, this script is re-executed once per GPU process (rank). 
    Args:
        None but reads the LOCAL_RANK and NODE_RANK environment variables set by Lightning's subprocess launcher on spawned (non rank-0) processes.
    Returns:
        bool: True if this is the rank-0 process, False otherwise.
    """
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


def record_model_parameters(dct_parameters: dict, bl_record: bool = True) -> int:
    """
    Record a model run to the ledger CSV.
    Checks rank to ensure that only the rank-0 process appends to the ledger under multi-GPU training.
    Args:
        dct_parameters (dict): A dictionary of parameters used for the model run (usually argparse).
        bl_record (bool): Whether to write to the ledger (should be True only for rank 0).
    Returns:
        The assigned run index (INT_MODEL), starting at 1.
    """
    pth_ledger: Path = Path(Directories.pth_data / "model_metadata.csv").resolve()
    df_ledger: pd.DataFrame = pd.read_csv(pth_ledger)

    int_model = 1
    if not df_ledger.empty and "INT_MODEL" in df_ledger.columns:
        int_model: int = pd.to_numeric(df_ledger["INT_MODEL"]).max()
        if bl_record:
            int_model += 1

    df_row = pd.DataFrame([{"INT_MODEL": str(int_model), "DCT_PARAMETERS": str(dct_parameters)}])
    if is_rank_zero() and bl_record:
        df_row.to_csv(pth_ledger, mode="a", header=False, index=False)
    return int_model


def save_results(results, int_model: int, str_model: str, str_class: str, str_split: str, pth_results: Path) -> None:
    """
    Saves the results of a model run to a CSV file (ensuring that only the rank-0 process writes to the file under multi-GPU training).
    Args:
        results (list[dict]): A list of dictionaries containing the results to be saved.
        int_model (int): The model run index (INT_MODEL).
        str_model (str): The model name (e.g., "classifier", "flexmatch", "lstm").
        str_class (str): The class type (e.g., "binary", "multi").
        str_split (str): The data split name (e.g., "train", "val", "test").
        pth_results (Path): The path to the results CSV file.
    Returns:
        None but appends the results to the specified CSV file.
    """
    if is_rank_zero():
        df = pd.DataFrame(results)

        lst_target_columns = ["int_model", "str_model", "str_class", "str_split"] + df.columns.tolist()
        if pth_results.exists():
            try:
                df_existing = pd.read_csv(pth_results)
                lst_target_columns: list[str] = df_existing.columns.tolist()
            except:
                logger.warning(f"Could not read existing results CSV at {pth_results}.")

        dct_column_values: dict = {
            "int_model": int_model,
            "str_model": str_model,
            "str_class": str_class,
            "str_split": str_split,
        }
        
        df_final = pd.DataFrame(index=range(len(df)))
        for col in lst_target_columns:
            if col in dct_column_values:
                df_final[col] = dct_column_values[col]
            elif col in df.columns:
                df_final[col] = df[col].values
            else:
                df_final[col] = ""
        
        df_final.to_csv(pth_results, mode="a", header=False, index=False, na_rep="")


def print_tensorboard_summary(pth_dir_child: Path, pth_dir_parent: Path = Directories.pth_logs) -> None:
    """
    Reads the first TensorBoard event file found under a directory and prints a summary of its logged scalars.
    Args:
        pth_dir_child (Path): Child directory under the parent directory to search for a "events.out.tfevents.*" file.
        pth_dir_parent (Path): Parent directory to search for a "events.out.tfevents.*" file (searched recursively,
            matching the first file in alphabetical/path order).
    Returns:
        None but logs the event file path and, for each logged scalar tag, the number of points and the
        first, last, minimum, and maximum values.
    Raises:
        FileNotFoundError: If no TensorBoard event file is found under pth_dir.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    lst_pth_events: list[Path] = sorted(Path(pth_dir_parent / pth_dir_child).glob("**/events.out.tfevents.*"))
    if not lst_pth_events:
        raise FileNotFoundError(f"No TensorBoard event file found under: {pth_dir_parent / pth_dir_child}")
    pth_event: Path = lst_pth_events[0]

    accumulator = EventAccumulator(str(pth_event.parent))
    accumulator.Reload()

    logger.info(f"TensorBoard event file: {pth_event}")
    lst_scalar_tags: list[str] = accumulator.Tags().get("scalars", [])
    if not lst_scalar_tags:
        logger.info("No scalar values were logged in this event file.")
        return

    for str_tag in lst_scalar_tags:
        lst_values: list[float] = [event.value for event in accumulator.Scalars(str_tag)]
        logger.info(
            f"  {str_tag}: n={len(lst_values)}, "
            f"first={lst_values[0]:.4f}, last={lst_values[-1]:.4f}, "
            f"min={min(lst_values):.4f}, max={max(lst_values):.4f}"
        )
