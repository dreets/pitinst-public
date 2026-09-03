import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pathlib import Path
from mmengine.dist import get_rank, is_distributed, barrier
from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore
from typing import cast

import utils
from models_classification import ClfInitialisers, BaseClassifier


class SegInitialisers:
    """Utility class for initialising segmentation model components (backbone, architecture, loss, metrics)."""
    @staticmethod
    def loss(int_classes: int) -> tuple[nn.Module, nn.Module]:
        if int_classes <= 2:
            loss_fn_ce: nn.Module = nn.BCEWithLogitsLoss()
            str_task = "binary"
        else:
            loss_fn_ce: nn.Module = nn.CrossEntropyLoss()
            str_task = "multiclass"

        loss_fn_dice: smp.losses.DiceLoss = smp.losses.DiceLoss(mode=str_task, from_logits=True)
        return loss_fn_ce, loss_fn_dice

    @staticmethod
    def metrics(int_classes: int) -> nn.ModuleDict:
        """Initialise a dictionary of segmentation metrics based on the number of classes."""
        if int_classes <= 2:
            dice = DiceScore(num_classes=2, average="macro", include_background=True, input_format="index")
        else:
            dice = DiceScore(num_classes=int_classes, average="macro", include_background=False, input_format="index")
        
        metrics_collection: MetricCollection = MetricCollection({"dice": dice})
        return nn.ModuleDict({str_split: metrics_collection.clone() for str_split in ["train_split", "val_split", "test_split"]})

    @staticmethod
    def metrics_binary() -> nn.ModuleDict:
        """Initialise a dictionary of binary Dice metrics for each split."""
        dice = DiceScore(num_classes=2, average="macro", include_background=True, input_format="index")
        metrics_collection: MetricCollection = MetricCollection({"dice_binary": dice})
        return nn.ModuleDict({str_split: metrics_collection.clone() for str_split in ["train_split", "val_split", "test_split"]})

    @staticmethod
    def architecture(str_architecture: str, str_backbone: str, int_classes: int = 2, pth_backbone_checkpoint: Path | None = None) -> nn.Module:
        """
        Create a segmentation model with optional pre-trained CNN backbone from a classifier checkpoint.
        Args:
            str_architecture: The segmentation architecture (e.g., "deeplabv3plus").
            str_backbone: The backbone encoder name (e.g., "resnet50").
            int_classes: Number of output classes (default: 2 for binary segmentation).
            str_backbone_checkpoint: Optional path to a pre-trained BaseClassifier checkpoint to extract the backbone from.
        Returns:
            nn.Module: The segmentation model with the specified encoder.
        """
        if str_backbone not in utils.Models.lst_cnnseg_encoders:
            raise ValueError(f"Unsupported encoder: {str_backbone}. Supported encoders: {utils.Models.lst_cnnseg_encoders}")

        dct_architectures = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
        }
        if str_architecture not in dct_architectures:
            raise ValueError(f"Unsupported architecture: {str_architecture}")

        model = smp.create_model(arch=str_architecture, encoder_name=str_backbone, classes=int_classes if int_classes > 2 else 1)
        
        if pth_backbone_checkpoint is not None: # load encoder weights if given
            checkpoint: dict = torch.load(str(pth_backbone_checkpoint), map_location="cpu")
            state_dict: dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            classifier: nn.Module = BaseClassifier(str_backbone=str_backbone, int_classes=int_classes)
            classifier.load_state_dict(state_dict, strict=False)
            encoder_state_dict = {}
            for key, value in classifier.backbone.state_dict().items():
                encoder_state_dict[key] = value
            model.encoder.load_state_dict(encoder_state_dict, strict=False)
        
        return model

    @staticmethod
    def backbone_mmseg(str_backbone: str, str_backbone_checkpoint: str | None = None) -> dict:
        dct_backbones = {
            "dinov2": ("TIMMBackbone", "vit_base_patch14_reg4_dinov2"),
        }
        if str_backbone not in dct_backbones:
            raise ValueError(f"Unsupported backbone: {str_backbone}")

        bl_backbone_checkpoint: bool = str_backbone_checkpoint is not None and Path(str_backbone_checkpoint).exists()
        if bl_backbone_checkpoint:
            str_backbone_checkpoint_new: str | None = SegInitialisers.remap_backbone_checkpoint_mmseg(str_backbone_checkpoint=str_backbone_checkpoint)
        else:
            str_backbone_checkpoint_new: str | None = None
        dct_backbone: dict=dict(
            type=dct_backbones[str_backbone][0],
            model_name=dct_backbones[str_backbone][1],
            features_only=True,
            pretrained=not bl_backbone_checkpoint,
            out_indices=(2, 5, 8, 11),
            init_cfg=dict(type="Pretrained", checkpoint=str_backbone_checkpoint_new)
        )
        return dct_backbone

    @staticmethod
    def remap_backbone_checkpoint_mmseg(str_backbone_checkpoint: str | None) -> str:
        """
        Renaming required for mmsegmentation to load timm model weights correctly.
        This function remaps the keys in the checkpoint to match the expected format for mmsegmentation.
        """
        if str_backbone_checkpoint is None:
            raise ValueError("str_backbone_checkpoint cannot be None")
        str_backbone_checkpoint_new: str = str(f"{utils.Directories.pth_models}/{Path(str_backbone_checkpoint).stem}_mm.pth")
        if get_rank() == 0: # avoid concurrent ranks torch.save()-ing to the same file
            checkpoint = torch.load(str_backbone_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            remapped_state_dict = {
                f"timm_model.model.{str_key[len('backbone.'):]}": tsr_weight
                for str_key, tsr_weight in state_dict.items()
                if str_key.startswith('backbone.')
            }
            torch.save(remapped_state_dict, str_backbone_checkpoint_new)
        if is_distributed():
            barrier()
        return str_backbone_checkpoint_new

    @staticmethod
    def neck_mmseg(str_architecture: str) -> dict:
        if str_architecture == "setrmla":
            return dict(type="MultiLevelNeck", in_channels=[768, 768, 768, 768], out_channels=256, scales=[1, 1, 1, 1])
        elif str_architecture in ["mask2former", "uperhead"]:
            return dict(type="MultiLevelNeck", in_channels=[768, 768, 768, 768], out_channels=256, scales=[4, 2, 1, 0.5])
        else:
            raise ValueError(f"Unsupported architecture: {str_architecture}")


class CNNSegmenter(L.LightningModule):
    """Segmentation model built on segmentation_models_pytorch architectures (default binary)."""
    def __init__(
            self,
            str_architecture: str,
            str_encoder: str,
            int_classes: int = 2,
            flt_lr: float = 1e-3,
            int_epochs_freeze: int = 10,
            pth_backbone_checkpoint: Path | None = None,
        ):
        """
        Args:
            str_architecture: The architecture of the segmentation model to initialise (e.g., "deeplabv3plus").
            str_encoder: The encoder to use for the encoder-decoder architecture (e.g., "resnet50").
            int_classes: Number of output classes (default: 2 for binary segmentation).
            flt_lr: Learning rate for the optimizer (default: 1e-3).
            int_epochs_freeze: Number of initial epochs to keep the backbone frozen before unfreezing it for fine-tuning (default: 10).
            pth_backbone_checkpoint: Optional path to a pre-trained BaseClassifier checkpoint to use as the backbone.
        """
        super().__init__()
        self.save_hyperparameters()
        self.int_classes: int = int_classes
        self.flt_lr: float = flt_lr
        self.int_epochs_freeze: int = int_epochs_freeze
        self.str_architecture: str = str_architecture
        self.str_encoder: str = str_encoder
        self.pth_backbone_checkpoint: Path | None = pth_backbone_checkpoint

        self.dct_metrics: nn.ModuleDict = SegInitialisers.metrics(int_classes=self.int_classes)
        self.dct_metrics_binary: nn.ModuleDict = SegInitialisers.metrics_binary()
        self.dct_metrics_clf: nn.ModuleDict = ClfInitialisers.metrics(int_classes=self.int_classes)
        self.dct_metrics_clf_binary: nn.ModuleDict = ClfInitialisers.metrics_binary()

        self.model = SegInitialisers.architecture(
            str_architecture=self.str_architecture,
            str_backbone=self.str_encoder,
            int_classes=self.int_classes,
            pth_backbone_checkpoint=self.pth_backbone_checkpoint,
        )
        self.backbone: nn.Module = self.model.encoder
        self.loss_fn_ce, self.loss_fn_dice = SegInitialisers.loss(int_classes=self.int_classes)
        self._freeze_backbone()

    def forward(self, tsr_images: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        tsr_logits = self.model(tsr_images)
        return tsr_logits

    def _shared_prediction_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Shared logic for prediction steps, returning values needed for loss and metric computation.
        Args:
            batch: A tuple containing a batch of images and their corresponding labels.
        Returns:
            A dictionary containing logits, labels, and predictions for loss and metric computation.
        """
        tsr_images, tsr_masks = batch
        tsr_preds_loss: torch.Tensor = self(tsr_images)

        if self.int_classes <= 2:
            tsr_masks_loss: torch.Tensor = tsr_masks.float()
            tsr_preds_multi: torch.Tensor = (torch.sigmoid(tsr_preds_loss) > 0.5).long()
            tsr_masks_multi: torch.Tensor = tsr_masks.long()
            tsr_preds_binary: torch.Tensor = tsr_preds_multi
            tsr_masks_binary: torch.Tensor = tsr_masks_multi
        else:
            tsr_masks_loss: torch.Tensor = tsr_masks.squeeze(1).long()
            tsr_preds_multi = torch.argmax(tsr_preds_loss, dim=1)
            tsr_masks_multi = tsr_masks_loss
            tsr_preds_binary: torch.Tensor = (tsr_preds_multi > 0).long().unsqueeze(1)
            tsr_masks_binary: torch.Tensor = (tsr_masks_multi > 0).long().unsqueeze(1)

        # Calculate frame-level classification metrics by determining the dominant class in each frame
        if self.int_classes <= 2:
            tsr_preds_classes: torch.Tensor = (tsr_preds_multi > 0.5).long().squeeze(1) # [B, H, W]
            tsr_masks_classes: torch.Tensor = tsr_masks_multi.squeeze(1) if tsr_masks_multi.ndim == 4 else tsr_masks_multi
        else:
            # tsr_preds_metric is already the per-pixel class-index map (argmax over channels done above)
            tsr_preds_classes: torch.Tensor = tsr_preds_multi # [B, H, W]
            tsr_masks_classes: torch.Tensor = tsr_masks_multi # [B, H, W]

        ls_batch_pred_classes: list = []
        ls_batch_mask_classes: list = []
        for i in range(tsr_preds_classes.shape[0]):
            tsr_pred_flat = tsr_preds_classes[i].flatten()
            tsr_mask_flat = tsr_masks_classes[i].flatten()

            tsr_pred_insts: torch.Tensor = tsr_pred_flat[tsr_pred_flat != 0] 
            if tsr_pred_insts.numel() > 0:
                tsr_pred_class: torch.Tensor = tsr_pred_insts.mode().values
            else:
                tsr_pred_class: torch.Tensor = torch.tensor(0, device=tsr_pred_flat.device, dtype=torch.long)
            
            tsr_mask_insts = tsr_mask_flat[tsr_mask_flat != 0]
            if tsr_mask_insts.numel() > 0:
                tsr_mask_class: torch.Tensor = tsr_mask_insts.mode().values
            else:
                tsr_mask_class: torch.Tensor = torch.tensor(0, device=tsr_mask_flat.device, dtype=torch.long)

            ls_batch_pred_classes.append(tsr_pred_class)
            ls_batch_mask_classes.append(tsr_mask_class)

        tsr_batch_preds: torch.Tensor = torch.stack(ls_batch_pred_classes)
        tsr_batch_masks: torch.Tensor = torch.stack(ls_batch_mask_classes)

        if self.int_classes <= 2:
            tsr_preds_binary_clf: torch.Tensor = tsr_batch_preds
            tsr_masks_binary_clf: torch.Tensor = tsr_batch_masks
            tsr_preds_multi_clf: torch.Tensor = tsr_preds_binary_clf
            tsr_masks_multi_clf: torch.Tensor = tsr_masks_binary_clf
        else:
            # Clamp indices to valid range [0, num_classes) to prevent out-of-bounds errors
            tsr_preds_multi_clf = torch.clamp(tsr_batch_preds.long(), 0, self.int_classes - 1)
            tsr_masks_multi_clf = torch.clamp(tsr_batch_masks.long(), 0, self.int_classes - 1)
            
            tsr_preds_binary_clf: torch.Tensor = (tsr_batch_preds > 0).long()
            tsr_masks_binary_clf: torch.Tensor = (tsr_batch_masks > 0).long()
            
        return {
            "tsr_preds_loss": tsr_preds_loss,
            "tsr_masks_loss": tsr_masks_loss,
            "tsr_preds_multi": tsr_preds_multi,
            "tsr_masks_multi": tsr_masks_multi,
            "tsr_preds_binary": tsr_preds_binary,
            "tsr_masks_binary": tsr_masks_binary,
            "tsr_preds_binary_clf": tsr_preds_binary_clf,
            "tsr_masks_binary_clf": tsr_masks_binary_clf,
            "tsr_preds_multi_clf": tsr_preds_multi_clf,
            "tsr_masks_multi_clf": tsr_masks_multi_clf,
        }

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], str_split: str) -> torch.Tensor:
        """
        Shared logic for training, validation, and testing steps. Computes the loss, Dice coefficient, and frame-level classification metrics.
        For each frame, the dominant (mode) class is determined from predictions and ground truth, and frame-level metrics are computed.
        Args:
            batch: A tuple containing a batch of images and their corresponding masks.
            str_split: The stage ("train_split", "val_split", or "test_split") for metric tracking.
        Returns:
            tsr_loss: The computed loss for the batch.
        """
        dct_output: dict[str, torch.Tensor] = self._shared_prediction_step(batch=batch)

        tsr_loss_ce: torch.Tensor = self.loss_fn_ce(dct_output["tsr_preds_loss"], dct_output["tsr_masks_loss"])
        tsr_loss_dice: torch.Tensor = self.loss_fn_dice(dct_output["tsr_preds_loss"], dct_output["tsr_masks_loss"])
        tsr_loss: torch.Tensor = tsr_loss_ce + tsr_loss_dice

        self.dct_metrics[str_split].update(dct_output["tsr_preds_multi"], dct_output["tsr_masks_multi"])
        self.dct_metrics_binary[str_split].update(dct_output["tsr_preds_binary"], dct_output["tsr_masks_binary"])

        self.dct_metrics_clf[str_split].update(dct_output["tsr_preds_multi_clf"], dct_output["tsr_masks_multi_clf"])
        self.dct_metrics_clf_binary[str_split].update(dct_output["tsr_preds_binary_clf"], dct_output["tsr_masks_binary_clf"])
        return tsr_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        non_backbone_params = [p for p in self.model.parameters() if p not in set(self.backbone.parameters())]
        return torch.optim.AdamW([
            {"params": non_backbone_params, "lr": self.flt_lr, "weight_decay": self.flt_lr * 10},
            {"params": self.backbone.parameters(), "lr": self.flt_lr * 0.01, "weight_decay": self.flt_lr * 0.1},
        ])

    def _log_metrics(self, str_split: str, batch: tuple[torch.Tensor, torch.Tensor], tsr_loss: torch.Tensor) -> None:
        """Log metrics for the specified split."""
        str_split_full: str = f"{str_split}_split"
        self.log(f"{str_split}_loss", tsr_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)

        metrics = cast(MetricCollection, self.dct_metrics[str_split_full])
        metrics_binary = cast(MetricCollection, self.dct_metrics_binary[str_split_full])
        clf_metrics = cast(MetricCollection, self.dct_metrics_clf[str_split_full])
        clf_metrics_binary = cast(MetricCollection, self.dct_metrics_clf_binary[str_split_full])

        self.log(f"{str_split}_dice", metrics["dice"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log(f"{str_split}_f1", clf_metrics["f1"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(metrics_binary, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(clf_metrics, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(clf_metrics_binary, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)

    def _shared_epoch_start(self, str_split: str) -> None:
        """Reset metrics at the start of each epoch for the specified split."""
        self.dct_metrics[f"{str_split}_split"].reset()
        self.dct_metrics_binary[f"{str_split}_split"].reset()
        self.dct_metrics_clf[f"{str_split}_split"].reset()
        self.dct_metrics_clf_binary[f"{str_split}_split"].reset()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        tsr_loss: torch.Tensor = self._shared_step(batch, "train_split")
        self._log_metrics(str_split="train", batch=batch, tsr_loss=tsr_loss)
        return tsr_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss: torch.Tensor = self._shared_step(batch, "val_split")
        self._log_metrics(str_split="val", batch=batch, tsr_loss=tsr_loss)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss: torch.Tensor = self._shared_step(batch, "test_split")
        self._log_metrics(str_split="test", batch=batch, tsr_loss=tsr_loss)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._shared_prediction_step(batch=batch)

    def on_train_epoch_start(self) -> None:
        """Reset train metrics and set backbone train/eval mode based on freeze schedule."""
        self._shared_epoch_start("train")

        if self._should_train_backbone():
            self.backbone.train()
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def on_validation_epoch_start(self) -> None:
        self._shared_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self._shared_epoch_start("test")

    def _should_train_backbone(self) -> bool:
        return self.current_epoch >= self.int_epochs_freeze

    def _freeze_backbone(self) -> None:
        if self.int_epochs_freeze > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()


def initialise_mmseg(str_architecture: str, int_classes: int) -> dict:
    """
    Initialise an mmsegmentation decode head config for the given architecture.
    All three options consume the same 4-level, 256-channel feature pyramid produced by MultiLevelNeck.
    """
    if str_architecture == "uperhead":
        return dict(
            type="UPerHead",
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=256,
            dropout_ratio=0.1,
            num_classes=int_classes,
            loss_decode=[dict(type="CrossEntropyLoss", loss_name="loss_ce"), dict(type="DiceLoss", loss_name="loss_dice")],
        )
    if str_architecture == "setrmla":
        return dict(
            type="SETRMLAHead",
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            mla_channels=128,
            channels=512, # must equal mla_channels * len(in_channels)
            dropout_ratio=0,
            num_classes=int_classes,
            loss_decode=[dict(type="CrossEntropyLoss", loss_name="loss_ce"), dict(type="DiceLoss", loss_name="loss_dice")],
        )
    
    if str_architecture == "mask2former":
        return dict(
            type="Mask2FormerHead",
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            in_index=[0, 1, 2, 3],
            num_classes=int_classes,
            num_things_classes=0,
            num_stuff_classes=int_classes,
            num_queries=100,
            pixel_decoder=dict(
                type="mmdet.MSDeformAttnPixelDecoder",
                num_outs=3,
                norm_cfg=dict(type="GN", num_groups=32),
                act_cfg=dict(type="ReLU"),
                encoder=dict(
                    num_layers=6,
                    layer_cfg=dict(
                        self_attn_cfg=dict(embed_dims=256, num_heads=8, num_levels=3, num_points=4, batch_first=True),
                        ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True))
                    )
                ),
                positional_encoding=dict(num_feats=128, normalize=True)
            ),
            enforce_decoder_input_project=False,
            positional_encoding=dict(num_feats=128, normalize=True),
            transformer_decoder=dict(
                return_intermediate=True,
                num_layers=9,
                layer_cfg=dict(
                    self_attn_cfg=dict(embed_dims=256, num_heads=8, batch_first=True),
                    cross_attn_cfg=dict(embed_dims=256, num_heads=8, batch_first=True),
                    ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True))
                )
            ),
            loss_cls=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=2.0, reduction="mean", class_weight=[1.0] * int_classes + [0.1]),
            loss_mask=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0),
            loss_dice=dict(type="mmdet.DiceLoss", use_sigmoid=True, activate=True, reduction="mean", naive_dice=True, eps=1.0, loss_weight=5.0),
            train_cfg=dict(
                num_points=12544, 
                oversample_ratio=3.0, 
                importance_sample_ratio=0.75,
                assigner=dict(
                    type="mmdet.HungarianAssigner",
                    match_costs=[
                        dict(type="mmdet.ClassificationCost", weight=2.0),
                        dict(type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                        dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0)
                    ]),
                sampler=dict(type="mmdet.MaskPseudoSampler")
            )
        )
    raise ValueError(f"Unsupported architecture: {str_architecture}")