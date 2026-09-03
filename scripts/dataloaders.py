import albumentations as album
import numpy as np
import pandas as pd
import shutil
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import mmcv
_str_mmcv_version: str = mmcv.__version__
mmcv.__version__ = "2.1.0" # installed mmcv (2.2.0) is API-compatible but newer than mmseg/mmdet's asserted upper bound
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
mmcv.__version__ = _str_mmcv_version

import utils

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PitInstDataset(Dataset):
    """
    Dataset for instrument classification and segmentation in video frames.
    Each sample returns an RGB frame and a mask represented by pixels, along with the frame's class index:
        If binary: 1 indicates the presence of an instrument polygon annotation, and 0 indicates background.
        If multiclass: The class index indicates the presence of the i-th instrument polygon annotation, and 0 indicates background.
    """
    def __init__(
        self,
        str_split: str,
        int_size: int = 224,
        bl_multiclass: bool = False,
        str_task: str = "clf",
        dct_split_videos: dict[str, list[str]] = utils.Videos.dct_split_videos,
        dct_str_int_class: dict[str, int] = utils.Instruments.dct_str_int_class,
        pth_frames: Path = utils.Directories.pth_frames,
        pth_annotations: Path = utils.Directories.pth_annotations,
    ):
        """
        Args:
            str_split: str, the dataset split (e.g., "train", "val", "test").
            int_size: int, the size to which images and masks will be resized (default: 224).
            bl_multiclass: bool, whether to use multiclass mode (default: False for binary).
            str_task: str, the task type (default: "clf", otherwise "seg").
            dct_split_videos: dict[str, list[str]], dictionary mapping splits to lists of video names.
            dct_str_int_class: dict[str, int], dictionary mapping instrument class names to integer indices.
            pth_frames: Path, path to the frames directory.
            pth_annotations: Path, path to the annotations directory.
        """
        self.str_split: str = str_split
        self.int_size: int = int_size
        self.bl_multiclass: bool = bl_multiclass
        self.str_task: str = str_task
        self.lst_videos: list[str] = dct_split_videos[str_split]
        self.dct_str_int_class: dict[str, int] = dct_str_int_class
        self.pth_frames: Path = pth_frames
        self.pth_annotations: Path = pth_annotations

        self.lst_frames: list[tuple[Path, int, list[tuple[int, int]]]] = self._lst_frames()
        self.transform: album.Compose = self._transforms()
        self.metainfo: dict = dict(classes=[str_class for str_class, _ in sorted(self.dct_str_int_class.items(), key=lambda item: item[1])])

    def _lst_frames(self) -> list[tuple[Path, int, list[tuple[int, int]]]]:
        """
        Build a list of frame paths and their corresponding annotation rows.
        Arguments:
            self.str_split: str, the dataset split (e.g., "train", "val", "test").
            self.dct_split_videos: dict[str, list[str]], dictionary mapping splits to lists of video names.
            self.pth_frames: Path, path to the frames directory.
            self.annotations_dir: Path, path to the annotations directory.
        Returns:
            Path: Path to frame image
            int: Class index of the first included instrument class present in the frame
            list[tuple[int, int]]: List of polygon coordinates for the instrument segmentation
        """
        lst_frames: list[tuple[Path, int, list[tuple[int, int]]]] = []
        for str_video in self.lst_videos:
            pth_frames_video: Path = self.pth_frames / str_video
            pth_annotations: Path = self.pth_annotations / f"{str_video}.csv"
            if not pth_frames_video.exists():
                raise FileNotFoundError(f"Frames directory for video '{str_video}' does not exist: {pth_frames_video}")
            if not pth_annotations.exists():
                raise FileNotFoundError(f"Annotations CSV for video '{str_video}' does not exist: {pth_annotations}")

            df_anno: pd.DataFrame = pd.read_csv(pth_annotations)
            lst_int_frames: list[int] = sorted(df_anno["INT_FRAME"].unique())
            lst_pth_frames: list[Path] = [pth_frames_video / f"{int_frame:06}.png" for int_frame in lst_int_frames if (pth_frames_video / f"{int_frame:06}.png").exists()]
            for pth_frame in lst_pth_frames:
                int_frame: int = int(pth_frame.stem)
                df_frame: pd.DataFrame = df_anno[df_anno["INT_FRAME"] == int_frame]
                df_frame_row: pd.Series | None = df_frame.iloc[0] if not df_frame.empty else None

                if df_frame_row is not None:
                    str_class: str = df_frame_row["STR_CLASS"]
                    if str_class in self.dct_str_int_class:
                        int_class: int = self.dct_str_int_class[str_class]
                        str_segmentation: str = df_frame_row["LST_SEGMENTATION"]
                        lst_coords: list[float] = [float(coord) for coord in str_segmentation.split(";")]
                        lst_polygon: list[tuple[int, int]] = [(int(lst_coords[i]), int(lst_coords[i + 1])) for i in range(0, len(lst_coords), 2)]

                        lst_frames.append((pth_frame, int_class, lst_polygon))
        return lst_frames

    def __len__(self) -> int:
        return len(self.lst_frames)

    def _build_mask(self, pth_frame: Path, int_class: int, lst_polygon: list[tuple[int, int]]) -> np.ndarray:
        """
        Create a segmentation mask from the frame's polygon annotations.
        If no valid polygon is present, the mask is all zeros (background).
        Args:
            pth_frame: Path to the frame image
            int_class: Class index of the instrument present in the frame
            lst_polygon: List of polygon coordinates for the instrument segmentation
            self.bl_multiclass: If True, the mask will contain class indices instead of binary values
        Returns:
            np.ndarray: Binary mask of shape (height, width) with 1s for instrument polygons and 0s elsewhere (or class indices if multiclass)
        Raises:
            ValueError: If the frame image cannot be opened or has invalid dimensions
        """
        img_img: Image.Image = Image.open(pth_frame).convert("RGB")
        width, height = img_img.size
        npy_mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)

        if len(lst_polygon) >= 3: # a valid polygon must have at least 3 points (excludes 0;0 no_instrument case)
            img_mask: Image.Image = Image.new("L", (width, height), 0)
            if self.bl_multiclass:
                ImageDraw.Draw(img_mask).polygon(lst_polygon, outline=int_class, fill=int_class)
            else:
                ImageDraw.Draw(img_mask).polygon(lst_polygon, outline=1, fill=1)
            npy_mask: np.ndarray = np.maximum(npy_mask, np.array(img_mask, dtype=np.uint8))
        return npy_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image] | None:
        """
        Returns a tuple of (image_tensor, mask_tensor) for a frame.
        Args:
            idx: int, index of the frame to retrieve
        Returns:
            torch.tensor: Image tensor [RGB, height, width]
            torch.tensor: Mask tensor [1, height, width] with 1s for instrument polygons and 0s elsewhere
        Raises:
            IndexError: If idx is out of bounds for the dataset
        """
        pth_frame, int_class, lst_polygon = self.lst_frames[idx]

        npy_mask: np.ndarray = self._build_mask(pth_frame, int_class, lst_polygon)
        npy_img: np.ndarray = np.array(Image.open(pth_frame).convert("RGB"), dtype=np.float32) / 255.0

        augmented = self.transform(image=npy_img, mask=npy_mask)
        tsr_img: torch.Tensor = augmented["image"]
        tsr_mask: torch.Tensor = augmented["mask"]
        tsr_mask = tsr_mask.unsqueeze(0)

        if self.str_task == "clf":
            return tsr_img, torch.tensor(int_class, dtype=torch.long)
        elif self.str_task == "seg":
            return tsr_img, tsr_mask
        else:
            raise ValueError(f"Unsupported task type: {self.str_task}. Supported types are 'clf' and 'seg'.")

    def _transforms(self) -> album.Compose:
        """
        Returns default train/val/test transforms for segmentation.
        Args:
            self.str_split: str, the split of the dataset ("train", "val", or "test").
            self.int_size: int, the size to which images and masks will be resized (default: 224).
        Returns:
            album.Compose: Composed transformations for the dataset split.
        """
        if self.str_split == "train":
            return album.Compose([
                album.Affine(),
                album.HorizontalFlip(),
                album.VerticalFlip(),
                album.Resize(height=self.int_size, width=self.int_size),
                album.ColorJitter(),
                album.pytorch.ToTensorV2(),
            ])
        else:
            return album.Compose([album.Resize(height=self.int_size, width=self.int_size), album.pytorch.ToTensorV2()])


def weighted_sampler(
        train_dataset,
        dct_str_int_class: dict[str, int] = utils.Instruments.dct_str_int_class,
        dct_volume_class: dict[str, int] = utils.Instruments.dct_volume_class,
    ) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that balances classification training batches according to the
    class frequencies recorded in class_metadata.csv (INT_TRAIN column).
    Args:
        train_dataset: PitInstDataset, the training dataset.
        dct_str_int_class: dict[str, int], mapping of instrument class names to integer indices (default: utils.Instruments.dct_str_int_class).
        dct_volume_class: dict[str, int], mapping of instrument class names to their volume (default: utils.Instruments.dct_volume_class).
    Returns:
        WeightedRandomSampler: Sampler assigning each sample a weight inversely proportional to frequency.
    Raises:
        ValueError: If a class present in dct_str_int_class has a recorded volume of 0.
    """
    dct_int_count: dict[int, int] = {int_class: dct_volume_class[str_class] for str_class, int_class in dct_str_int_class.items()}
    if any(int_count == 0 for int_count in dct_int_count.values()):
        raise ValueError("Cannot build a weighted sampler: at least one class has an INT_TRAIN count of 0.")

    dct_int_weight: dict[int, float] = {int_class: 1.0 / int_count for int_class, int_count in dct_int_count.items()}
    lst_weights: list[float] = [dct_int_weight[int_class] for _, int_class, _ in train_dataset.lst_frames]

    return WeightedRandomSampler(weights=lst_weights, num_samples=len(lst_weights), replacement=True)


def initialise_dataloaders(
    str_task: str = "clf",
    bl_multiclass: bool = False,
    int_batch: int = 16,
    int_workers: int = 4,
    int_size: int = 224,
    bl_shuffle: bool = True,
    bl_weighted_sampler: bool = True,
    bl_mmseg: bool = False,
    int_clip: int = 0,
    dct_split_videos: dict[str, list[str]] = utils.Videos.dct_split_videos,
    dct_str_int_class: dict[str, int] = utils.Instruments.dct_str_int_class,
    dct_volume_class: dict[str, int] = utils.Instruments.dct_volume_class,
    pth_frames: Path = utils.Directories.pth_frames,
    pth_annotations: Path = utils.Directories.pth_annotations,
) -> dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders for classification or segmentation.
    Args:
        str_task: str, the task type (default: "clf"). Supports "clf" and "seg".
        bl_multiclass: bool, whether to use multiclass mode (default: False for binary).
        int_batch: int, batch size for the dataloaders (default: 16).
        int_workers: int, number of worker processes for data loading (default: 4).
        int_size: int, size to which images and masks will be resized (default: 224).
        bl_shuffle: bool, whether to shuffle the training dataloader (default: True).
            Ignored if int_clip > 0 (clip mode disables shuffling) or if bl_weighted_sampler is False.
        bl_weighted_sampler: bool, whether to use the weighted sampler even if it could be used (default: True).
        int_clip: int, clip size (number of consecutive frames per sample) for temporal LSTM training.
            Only used with str_task == "classification". Default: 0 (single-frame mode).
        bl_mmseg: bool, whether to use mmsegmentation's collate function (default: False). Only used for segmentation.
        dct_split_videos: dict[str, list[str]], dictionary mapping splits to lists of video names.
        dct_str_int_class: dict[str, int], dictionary mapping instrument class names to integer indices.
        dct_volume_class: dict[str, int], dictionary mapping instrument class names to their INT_TRAIN
                    frame counts (from class_metadata.csv), used to build the weighted sampler.
        pth_frames: Path, path to the frames directory.
        pth_annotations: Path, path to the annotations directory.
    Returns:
        dict[str, DataLoader]: Dictionary containing "train", "val", and "test" dataloaders.
    Raises:
        ValueError: If int_clip > 0 but str_task != "clf".
        ValueError: If bl_weighted_sampler is True but str_task != "clf" or int_clip > 0.
    """
    if int_clip > 0 and str_task != "clf":
        raise ValueError(
            f"Temporal clip mode (int_clip={int_clip}) is only supported for classification tasks, "
            f"but str_task='{str_task}' was specified."
        )
    if not bl_weighted_sampler and int_clip > 0:
        raise ValueError("Weighted sampler is not supported in temporal clip mode (int_clip > 0).")
    
    dct_datasets: dict[str, Dataset] = {
        str_split: PitInstDataset(
            str_split=str_split,
            str_task=str_task,
            bl_multiclass=bl_multiclass,
            dct_split_videos=dct_split_videos,
            int_size=int_size,
            dct_str_int_class=dct_str_int_class,
            pth_frames=pth_frames,
            pth_annotations=pth_annotations,
        )
        for str_split in ["train", "val", "test"]
    }

    if str_task == "clf" and int_clip > 0: # i.e. temporal clip mode for LSTM training
        collate_fn = lambda batch: collate_temporal_clip(batch, int_clip) # stack consecutive frames into temporal clips
        sampler = None # weighted sampler is not supported in temporal clip mode
        shuffle = False # shuffling is disabled in temporal clip mode to preserve frame order
    elif str_task == "seg" and bl_mmseg:
        collate_fn = collate_mmseg
        sampler = weighted_sampler(dct_datasets["train"], dct_str_int_class, dct_volume_class) if bl_weighted_sampler else None
        shuffle = bl_shuffle and (not bl_weighted_sampler) # shuffling is disabled if weighted sampler is used
    else:
        collate_fn = None
        sampler = weighted_sampler(dct_datasets["train"], dct_str_int_class, dct_volume_class) if bl_weighted_sampler else None
        shuffle = bl_shuffle and (not bl_weighted_sampler) # shuffling is disabled if weighted sampler is used

    dct_dataloaders: dict[str, DataLoader] = {
        str_split: DataLoader(
            dataset=dct_datasets[str_split],
            batch_size=int_batch,
            shuffle=shuffle if str_split == "train" else False,
            sampler=sampler if str_split == "train" else None,
            num_workers=int_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for str_split in ["train", "val", "test"]
    }
    return dct_dataloaders


def initialise_video_dataloaders(
    str_task: str = "clf",
    bl_multiclass: bool = False,
    int_batch: int = 16,
    int_workers: int = 4,
    int_size: int = 224,
    lst_videos: list[str] = utils.Videos.lst_val + utils.Videos.lst_test,
    dct_str_int_class: dict[str, int] = utils.Instruments.dct_str_int_class,
    pth_frames: Path = utils.Directories.pth_frames,
    pth_annotations: Path = utils.Directories.pth_annotations,
) -> dict[str, DataLoader]:
    """
    Create individual dataloaders for each video without shuffling, weighted sampling, or splits.
    Args:
        str_task: str, the task type (default: "clf"). Supports "clf" and "seg".
        bl_multiclass: bool, whether to use multiclass mode (default: False for binary).
        int_batch: int, batch size for the dataloaders (default: 16).
        int_workers: int, number of worker processes for data loading (default: 4).
        int_size: int, size to which images and masks will be resized (default: 224).
        lst_videos: list[str], list of all video names.
        dct_str_int_class: dict[str, int], dictionary mapping instrument class names to integer indices.
        pth_frames: Path, path to the frames directory.
        pth_annotations: Path, path to the annotations directory.
    Returns:
        dict[str, DataLoader]: Dictionary with video names as keys and dataloaders as values.
    """
    dct_video_dataloaders: dict[str, DataLoader] = {}
    pth_annotations_temp: Path = pth_annotations / "temp"
    pth_annotations_temp.mkdir(parents=True, exist_ok=True)
    for str_video in lst_videos:
        pth_annotations_video: Path = pth_annotations / f"{str_video}.csv"
        pth_annotations_temp_video: Path = pth_annotations_temp / f"{str_video}.csv"
        if pth_annotations_video.exists():
            shutil.copy2(pth_annotations_video, pth_annotations_temp_video)
        else:
            pth_video_frames: Path = pth_frames / str_video
            if not pth_video_frames.exists():
                raise FileNotFoundError(f"Frames directory for video '{str_video}' does not exist: {pth_video_frames}.")
            lst_frame_files: list[Path] = sorted(pth_video_frames.glob("*.png"))
            df_empty: pd.DataFrame = pd.DataFrame([])
            for pth_frame in lst_frame_files:
                df_frame: pd.DataFrame = pd.DataFrame([{
                    "STR_VIDEO": str_video,
                    "INT_FRAME": int(pth_frame.stem),
                    "STR_CLASS": "no_instrument",
                    "LST_SEGMENTATION": "0;0"
                }])
                df_empty = pd.concat([df_empty, df_frame], ignore_index=True)
            df_empty.to_csv(pth_annotations_temp_video, index=False)
   
        video_dataset: Dataset = PitInstDataset(
            str_split="video", # "video" is a dummy split
            str_task=str_task,
            bl_multiclass=bl_multiclass,
            dct_split_videos={"video": [str_video]}, # dictionary with a single video for this dataset
            int_size=int_size,
            dct_str_int_class=dct_str_int_class,
            pth_frames=pth_frames,
            pth_annotations=pth_annotations_temp,
        )
        
        dct_video_dataloaders[str_video] = DataLoader(
            dataset=video_dataset,
            batch_size=int_batch,
            shuffle=False,
            sampler=None,
            num_workers=int_workers,
            pin_memory=True,
        )

    shutil.rmtree(pth_annotations_temp) # clean up temporary annotations directory
    return dct_video_dataloaders


def collate_temporal_clip(batch: list, int_clip: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for temporal clip training.
    Stacks consecutive frames from the batch into temporal sequences.
    Each clip contains int_clip consecutive frames, with the label as the mode (most common label) across all frames in the clip.
    Note: collates the end of a video to the next video, but this is acceptable for training for this classification task.
    Args:
        batch: List of (image, label) tuples from the dataset.
        int_clip: Number of consecutive frames to stack per clip.
    
    Returns:
        Stacked_images: Tensor of shape [num_clips, int_clip, 3, H, W]
        Labels: Tensor of shape [num_clips]
    """
    lst_clips: list[torch.Tensor] = []
    lst_labels: list[torch.Tensor] = []
    
    # Group batch items into clips of size int_clip, stack frames along new dimension: [int_clip, 3, H, W]
    for i in range(0, len(batch), int_clip):
        clip_items = batch[i : i + int_clip]
        
        if len(clip_items) < int_clip:
            last_frame, last_label = clip_items[-1]
            while len(clip_items) < int_clip:
                clip_items.append((last_frame, last_label))
        
        tsr_clip = torch.stack([frame for frame, _ in clip_items], dim=0)
        label = torch.mode(torch.stack([label for _, label in clip_items])).values # modal label
        lst_clips.append(tsr_clip)
        lst_labels.append(label)
    
    if lst_clips:
        tsr_images = torch.stack(lst_clips, dim=0) # [num_clips, int_clip, 3, H, W]
        tsr_labels = torch.stack(lst_labels) # [num_clips]
        return tsr_images, tsr_labels
    
    return torch.tensor([]), torch.tensor([]) # fallback for edge case where no complete clips exist


def collate_mmseg(batch):
    """
    Custom collate function for mmsegmentation training.
    Stacks images and constructs SegDataSample objects for each sample in the batch.
    Args:
        batch: List of (image, mask) tuples from the dataset.
    Returns:
        dict: Dictionary containing 'inputs' (stacked images) and 'data_samples' (list of SegDataSample objects).
    """
    imgs = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    
    batch_inputs = torch.stack(imgs, dim=0)
    
    data_samples = []
    for img, mask in zip(imgs, masks):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        sample = SegDataSample()
        sample.gt_sem_seg = PixelData(data=mask.long())
        tpl_shape = tuple(img.shape[-2:])
        sample.set_metainfo({
            "ori_shape": tpl_shape,
            "img_shape": tpl_shape,
            "pad_shape": tpl_shape,
            "padding_size": [0, 0, 0, 0],
        })
        data_samples.append(sample)
        
    return {'inputs': batch_inputs, 'data_samples': data_samples}


class UnlabelledFrameDataset(Dataset):
    """
    Dataset of unlabelled video frames for FlexMatch semi-supervised classification training.
    Frames are drawn from utils.Videos.lst_ss_train (STR_SPLIT == "ss_train"), which have no instrument annotations. 
    Each sample returns its own dataset index alongside a weakly-augmented view (used to generate the pseudo-label) 
    and a strongly-augmented view (used for the consistency loss against that pseudo-label), following the FixMatch/FlexMatch recipe.
    """
    def __init__(
        self,
        lst_videos: list[str] = utils.Videos.lst_ss_train,
        int_size: int = 224,
        pth_frames: Path = utils.Directories.pth_frames,
    ):
        """
        Args:
            lst_videos: list[str], unlabelled video names to draw frames from (default: utils.Videos.lst_ss_train).
            int_size: int, the size to which frames will be resized (default: 224).
            pth_frames: Path, path to the frames directory.
        """
        self.lst_videos: list[str] = lst_videos
        self.int_size: int = int_size
        self.pth_frames: Path = pth_frames

        self.lst_pth_frames: list[Path] = self._lst_frames()
        self.transform_weak, self.transform_strong = self._transforms()

    def _lst_frames(self) -> list[Path]:
        """
        Collects every frame path for the unlabelled videos.
        Args:
            self.lst_videos: list[str], unlabelled video names.
            self.pth_frames: Path, path to the frames directory.
        Returns:
            list[Path]: Sorted list of frame paths across all requested videos (grouped by video).
        Raises:
            FileNotFoundError: If no frames are found across any of the requested videos.
        """
        lst_pth_frames: list[Path] = []
        for str_video in self.lst_videos:
            pth_frames_video: Path = self.pth_frames / str_video
            if not pth_frames_video.exists():
                logger.warning(f"Frames directory not found for unlabelled video: {str_video}")
                continue
            lst_pth_frames.extend(sorted(pth_frames_video.glob("*.png")))

        if not lst_pth_frames:
            raise FileNotFoundError(f"No unlabelled frames found for videos: {self.lst_videos}")
        return lst_pth_frames

    def __len__(self) -> int:
        return len(self.lst_pth_frames)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Returns the dataset index and weak/strong augmented views of a frame.
        Args:
            idx: int, index of the frame to retrieve.
        Returns:
            int: The dataset index of the frame (used to track per-sample curriculum state in FlexMatch).
            torch.Tensor: Weakly-augmented image tensor [RGB, height, width].
            torch.Tensor: Strongly-augmented image tensor [RGB, height, width].
        """
        pth_frame: Path = self.lst_pth_frames[idx]
        npy_img: np.ndarray = np.array(Image.open(pth_frame).convert("RGB"), dtype=np.float32) / 255.0

        tsr_weak: torch.Tensor = self.transform_weak(image=npy_img)["image"]
        tsr_strong: torch.Tensor = self.transform_strong(image=npy_img)["image"]
        return idx, tsr_weak, tsr_strong

    def _transforms(self) -> tuple[album.Compose, album.Compose]:
        """
        Builds the weak and strong augmentation pipelines used by FlexMatch.
        Weak augmentation mirrors standard geometric augmentation (flip only, no photometric
        distortion). Strong augmentation additionally applies heavier photometric/geometric
        distortion and cutout, following the FixMatch/FlexMatch strong-augmentation recipe.
        Args:
            self.int_size: int, the size to which frames will be resized.
        Returns:
            album.Compose: Weak augmentation pipeline.
            album.Compose: Strong augmentation pipeline.
        """
        transform_weak: album.Compose = album.Compose([
            album.Resize(height=self.int_size, width=self.int_size),
            album.HorizontalFlip(),
            album.pytorch.ToTensorV2(),
        ])
        transform_strong: album.Compose = album.Compose([
            album.Resize(height=self.int_size, width=self.int_size),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Affine(),
            album.ColorJitter(),
            album.GaussianBlur(p=0.5),
            album.CoarseDropout(p=0.5),
            album.pytorch.ToTensorV2(),
        ])
        return transform_weak, transform_strong


def initialise_flexmatch_dataloaders(
    bl_multiclass: bool = False,
    int_batch: int = 64,
    int_mu: int = 7,
    int_workers: int = 4,
    int_size: int = 224,
    bl_shuffle: bool = True,
    bl_weighted_sampler: bool = True,
    dct_split_videos: dict[str, list[str]] = utils.Videos.dct_split_videos,
    dct_str_int_class: dict[str, int] = utils.Instruments.dct_str_int_class,
    pth_frames: Path = utils.Directories.pth_frames,
    pth_annotations: Path = utils.Directories.pth_annotations,
    lst_ss_videos: list[str] = utils.Videos.lst_ss_train,
) -> dict[str, DataLoader]:
    """
    Create dataloaders for FlexMatch semi-supervised classification training.
    Combines the standard labelled classification dataloaders with an additional unlabelled
    dataloader drawn from lst_ss_videos (default: utils.Videos.lst_ss_train).
    Args:
        bl_multiclass: bool, whether to use multiclass mode (default: False for binary).
        int_batch: int, batch size for the labelled dataloaders (default: 64).
        int_mu: int, ratio of unlabelled to labelled batch size, following the FixMatch/FlexMatch
            convention of oversampling unlabelled data relative to labelled data (default: 7).
        int_workers: int, number of worker processes for data loading (default: 4).
        int_size: int, size to which images will be resized (default: 224).
        bl_shuffle: bool, whether to shuffle the labelled training dataloader (default: True).
        bl_weighted_sampler: bool, whether to use the weighted sampler for the labelled training dataloader (default: True).
        dct_split_videos: dict[str, list[str]], dictionary mapping splits to lists of video names.
        dct_str_int_class: dict[str, int], dictionary mapping instrument class names to integer indices.
        pth_frames: Path, path to the frames directory.
        pth_annotations: Path, path to the annotations directory.
        lst_ss_videos: list[str], unlabelled video names for FlexMatch (default: utils.Videos.lst_ss_train).
    Returns:
        dict[str, DataLoader]: Dictionary with "train" (labelled), "train_unlabeled", "val", and "test" dataloaders.
    """
    dct_dataloaders: dict[str, DataLoader] = initialise_dataloaders(
        str_task="clf",
        bl_multiclass=bl_multiclass,
        int_batch=int_batch,
        int_workers=int_workers,
        int_size=int_size,
        bl_shuffle=bl_shuffle,
        bl_weighted_sampler=bl_weighted_sampler,
        dct_split_videos=dct_split_videos,
        dct_str_int_class=dct_str_int_class,
        pth_frames=pth_frames,
        pth_annotations=pth_annotations,
    )
    unlabeled_dataset: Dataset = UnlabelledFrameDataset(
        lst_videos=lst_ss_videos,
        int_size=int_size,
        pth_frames=pth_frames,
    )
    dct_dataloaders["train_unlabeled"] = DataLoader(
        unlabeled_dataset,
        batch_size=int_batch * int_mu,
        shuffle=True,
        num_workers=int_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dct_dataloaders