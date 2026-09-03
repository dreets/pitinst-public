import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import FocalLoss
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from typing import cast

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClfInitialisers:
    """Utility class for initialising classifier model components (backbone, loss, metrics, dropout)."""
    @staticmethod
    def backbone(str_backbone: str) -> tuple[nn.Module, int]:
        """
        Build a feature-extractor backbone for the given architecture.
        torchvision models are not used as they do not contain more modern weights for e.g. ResNet50.
        Args:
            str_backbone: The backbone to build (e.g. "resnet50").
        Returns:
            nn.Module: The backbone feature extractor.
            int: The dimensionality of the backbone's output features.
        Raises:
            ValueError: If the backbone is not supported.
        """
        if str_backbone == "moco":
            return moco_backbone(checkpoint_path="models/moco_surg.torch")
        
        dct_models: dict[str, tuple[str, int]] = {
            "convnextv2": ("convnextv2_tiny.fcmae_ft_in22k_in1k", 768),
            "densenet121": ("densenet121.ra_in1k", 1024),
            "dinov2": ("vit_base_patch14_reg4_dinov2", 768),
            "efficientnetv2": ("tf_efficientnetv2_s.in21k_ft_in1k", 1280),
            "resnet18": ("resnet18.a1_in1k", 512),
            "resnet50": ("resnet50.a1_in1k", 2048),
            "resnet101": ("resnet101.a1_in1k", 2048),
            "resnet152": ("resnet152.a1_in1k", 2048),
            "swin": ("swin_small_patch4_window7_224.ms_in22k_ft_in1k", 768),
        }
        if str_backbone not in dct_models:
            raise ValueError(f"Unsupported backbone: {str_backbone}")

        str_timm_name, int_feature_dim = dct_models[str_backbone]
        backbone = timm.create_model(str_timm_name, pretrained=True, num_classes=0)
        return backbone, int_feature_dim

    @staticmethod
    def loss(int_classes: int, str_loss: str = "ce") -> nn.Module:
        """
        Initialise the appropriate loss function based on the number of classes.
        Args:
            int_classes: Number of output classes (default: 2 for binary classification).
            str_loss: The type of loss function to use ("bce" always for binary, "ce" or "focal" for multi-class).
        Returns:
            nn.Module: The loss function.
        """
        if int_classes <= 2:
            return nn.BCEWithLogitsLoss()
        else:
            if str_loss == "ce":
                return nn.CrossEntropyLoss()
            elif str_loss == "focal":
                return FocalLoss(mode="multiclass", alpha=0.5, gamma=2.0)
            else:
                raise ValueError(f"Unsupported loss type for multi-class classification: {str_loss}")

    @staticmethod
    def metrics(int_classes: int) -> nn.ModuleDict:
        """Initialise a dictionary of classification metrics based on the number of classes."""
        if int_classes <= 2:
            metrics_collection: MetricCollection = MetricCollection({
                "accuracy": Accuracy(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
                "f1_macro": F1Score(task="binary", average="macro"),
                "f1": F1Score(task="binary"),
            })
        else:
            metrics_collection: MetricCollection = MetricCollection({
                "accuracy": Accuracy(task="multiclass", num_classes=int_classes, average="micro"),
                "precision": Precision(task="multiclass", num_classes=int_classes, average="macro"),
                "recall": Recall(task="multiclass", num_classes=int_classes, average="macro"),
                "f1_macro": F1Score(task="multiclass", num_classes=int_classes, average="macro"),
                "f1": F1Score(task="multiclass", num_classes=int_classes, average="macro", ignore_index=0), 
            })
            
        return nn.ModuleDict({str_split: metrics_collection.clone() for str_split in ["train_split", "val_split", "test_split"]})

    @staticmethod
    def metrics_binary() -> nn.ModuleDict:
        """Creates binary frame-level classification metrics (Any Foreground vs. Background)."""
        metrics_collection = MetricCollection({"f1_binary": F1Score(task="binary")})
        return nn.ModuleDict({str_split: metrics_collection.clone() for str_split in ["train_split", "val_split", "test_split"]})
    
    @staticmethod
    def dropout(bl_dropout: bool, int_in_features: int, int_out_features: int) -> nn.Module:
        """
        Initialise a classifier head with optional dropout.
        Args:
            bl_dropout: Whether to include a dropout layer before the linear layer.
            int_in_features: Number of input features to the linear layer.
            int_out_features: Number of output features from the linear layer.
        Returns:
            nn.Module: The classifier head (with or without dropout).
        """
        if bl_dropout:
            return nn.Sequential(nn.Dropout(p=0.5), nn.Linear(int_in_features, int_out_features))
        else:
            return nn.Linear(int_in_features, int_out_features)


class BaseClassifier(L.LightningModule):
    """Shared training and evaluation logic for classification models (default binary)."""
    def __init__(
            self,
            str_backbone: str,
            int_classes: int = 2,
            flt_lr: float = 1e-4,
            int_epochs_freeze: int = 5,
            bl_dropout: bool = True,
            str_loss: str = "ce",
        ):
        """
        Args:
            str_backbone: The backbone of the classification model to initialise (e.g., "resnet50").
            int_classes: Number of output classes (default: 2 for binary classification).
            flt_lr: Learning rate for the optimizer (default: 1e-4).
            int_epochs_freeze: Number of initial epochs to keep the backbone frozen before unfreezing it for fine-tuning (default: 5)
            bl_dropout: Whether to enable dropout before the classifier head (default: True).
            str_loss: The type of loss function to use in multiclass (default: "ce" for cross-entropy; "focal" for focal loss).
        """
        super().__init__()
        self.save_hyperparameters()
        self.int_classes: int = int_classes
        self.flt_lr: float = flt_lr
        self.int_epochs_freeze: int = int_epochs_freeze
        self.bl_dropout: bool = bl_dropout
        self.str_loss: str = str_loss

        self.backbone, self.int_backbone_dim = ClfInitialisers.backbone(str_backbone)
        self.fc: nn.Module = ClfInitialisers.dropout(bl_dropout=self.bl_dropout,int_in_features=self.int_backbone_dim, int_out_features=1 if int_classes <= 2 else int_classes)
        self.loss_fn: nn.Module = ClfInitialisers.loss(int_classes=self.int_classes, str_loss=self.str_loss)
        self.dct_metrics: nn.ModuleDict = ClfInitialisers.metrics(int_classes=self.int_classes)
        self.dct_metrics_binary: nn.ModuleDict = ClfInitialisers.metrics_binary()
        
        if self.int_epochs_freeze > 0: # freeze the backbone for  int_epochs_freeze epochs
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, tsr_images: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        assert self.fc is not None
        tsr_features = self.backbone(tsr_images)
        return self.fc(tsr_features.flatten(start_dim=1))

    def _shared_prediction_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Shared logic for prediction steps, returning values needed for loss and metric computation.
        Args:
            batch: A tuple containing a batch of images and their corresponding labels.
        Returns:
            A dictionary containing logits, labels, and predictions for loss and metric computation.
        """
        tsr_images, tsr_labels = batch
        tsr_logits: torch.Tensor = self(tsr_images)

        if self.int_classes <= 2:
            tsr_preds_loss: torch.Tensor = tsr_logits.view(-1).float()
            tsr_labels_loss: torch.Tensor = (tsr_labels.view(-1) > 0).float()
            tsr_preds_multi: torch.Tensor = (torch.sigmoid(tsr_preds_loss) > 0.5).long()
            tsr_labels_multi: torch.Tensor = (tsr_labels.view(-1) > 0).long()
            tsr_preds_binary: torch.Tensor = tsr_preds_multi
            tsr_labels_binary: torch.Tensor = tsr_labels_multi
        else:
            tsr_preds_loss: torch.Tensor = tsr_logits.float()
            tsr_labels_loss: torch.Tensor = tsr_labels.view(-1).long()
            tsr_preds_multi: torch.Tensor = torch.argmax(tsr_logits, dim=1)
            tsr_labels_multi: torch.Tensor = tsr_labels.view(-1).long()
            tsr_preds_binary: torch.Tensor = (tsr_preds_multi > 0).long()
            tsr_labels_binary: torch.Tensor = (tsr_labels_multi > 0).long()

        return {
            "tsr_preds_loss": tsr_preds_loss,
            "tsr_labels_loss": tsr_labels_loss,
            "tsr_preds_multi": tsr_preds_multi,
            "tsr_labels_multi": tsr_labels_multi,
            "tsr_preds_binary": tsr_preds_binary,
            "tsr_labels_binary": tsr_labels_binary,
        }

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], str_split: str) -> torch.Tensor:
        """
        Shared logic for training, validation, and testing steps. Computes the loss and classification metrics.
        Args:
            batch: A tuple containing a batch of images and their corresponding labels.
            str_split: The stage ("train_split", "val_split", or "test_split") for metric tracking.
        Returns:
            tsr_loss: The computed loss for the batch.
        """
        dct_output: dict[str, torch.Tensor] = self._shared_prediction_step(batch=batch)
        tsr_loss: torch.Tensor = self.loss_fn(dct_output["tsr_preds_loss"], dct_output["tsr_labels_loss"])
        self.dct_metrics[str_split].update(dct_output["tsr_preds_multi"], dct_output["tsr_labels_multi"])
        self.dct_metrics_binary[str_split].update(dct_output["tsr_preds_binary"], dct_output["tsr_labels_binary"])
        return tsr_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([
            {"params": self.fc.parameters(), "lr": self.flt_lr, "weight_decay": self.flt_lr * 10},
            {"params": self.backbone.parameters(), "lr": self.flt_lr * 0.1, "weight_decay": self.flt_lr},
        ])

    def _log_metrics(self, str_split: str, batch: tuple[torch.Tensor, torch.Tensor], tsr_loss: torch.Tensor) -> None:
        """Log metrics for the given split (train, val, or test)."""
        str_split_full: str = f"{str_split}_split"
        self.log(f"{str_split}_loss", tsr_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        metrics = cast(MetricCollection, self.dct_metrics[str_split_full])
        clf_metrics_binary = cast(MetricCollection, self.dct_metrics_binary[str_split_full])
        self.log(f"{str_split}_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log(f"{str_split}_f1_binary", clf_metrics_binary["f1_binary"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(clf_metrics_binary, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        tsr_loss = self._shared_step(batch, "train_split")
        self._log_metrics(str_split="train", batch=batch, tsr_loss=tsr_loss)
        return tsr_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._shared_step(batch, "val_split")
        self._log_metrics(str_split="val", batch=batch, tsr_loss=tsr_loss)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._shared_step(batch, "test_split")
        self._log_metrics(str_split="test", batch=batch, tsr_loss=tsr_loss)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._shared_prediction_step(batch=batch)

    def _shared_epoch_start(self, str_split: str) -> None:
        """Reset metrics at the start of each epoch for the specified split."""
        self.dct_metrics[f"{str_split}_split"].reset()
        self.dct_metrics_binary[f"{str_split}_split"].reset()
    
    def on_train_epoch_start(self) -> None:
        """Set backbone train/eval mode based on freeze schedule."""
        self._shared_epoch_start("train")

        if self.backbone is None:
            return
        
        if self._should_train_backbone():
            self.backbone.train()
            for param in self.backbone.parameters(): # unfreeze backbone parameters
                param.requires_grad = True
        else:
            self.backbone.eval()
            for param in self.backbone.parameters(): # freeze backbone parameters
                param.requires_grad = False

    def on_validation_epoch_start(self) -> None:
        self._shared_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self._shared_epoch_start("test")

    def _should_train_backbone(self) -> bool:
        return self.current_epoch >= self.int_epochs_freeze


class FlexMatchClassifier(L.LightningModule):
    """
    FlexMatch semi-supervised classifier. 
    For each unlabelled frame, a pseudo-label is predicted from its weakly-augmented view; 
    that pseudo-label supervises a cross-entropy loss on the strongly-augmented view,
    but only if the model's confidence exceeds a per-class threshold.

    Expects training batches shaped like Lightning's CombinedLoader output when combining a labelled
    and an unlabelled dataloader under the keys "labeled" and "unlabeled", i.e.:
        batch = {"labeled": (tsr_images, tsr_labels), "unlabeled": (tsr_idx, tsr_images_weak, tsr_images_strong)}
    Validation/test batches are the standard (tsr_images, tsr_labels) tuples (labelled data only).
    """
    def __init__(
            self,
            str_backbone: str = "resnet50",
            int_classes: int = 2,
            int_unlabeled: int = 1,
            flt_lr: float = 1e-3,
            flt_threshold: float = 0.95,
            flt_lambda_u: float = 1.0,
            int_epochs_freeze: int = 5,
            bl_dropout: bool = True,
        ):
        """
            str_backbone: Backbone to use (e.g. "resnet50").
            int_unlabeled: Size of the unlabelled dataset, used to size the per-sample curriculum memory buffer
                (see dataloaders.UnlabelledFrameDataset / dataloaders.initialise_flexmatch_dataloaders).
            int_classes: Number of output classes (default: 2 for binary classification).
            flt_lr: Learning rate for the optimizer.
            flt_threshold: Base confidence threshold (tau) for pseudo-labelling (default: 0.95).
            flt_lambda_u: Weight of the unsupervised (unlabelled) consistency loss term (default: 1.0).
            int_epochs_freeze: Number of initial epochs to keep the backbone frozen before unfreezing it for fine-tuning (default: 5).
            bl_dropout: Whether to apply dropout before the classifier head (default: True).
        """
        super().__init__()
        self.save_hyperparameters()
        self.int_classes: int = int_classes
        self.flt_lr: float = flt_lr
        self.flt_threshold: float = flt_threshold
        self.flt_lambda_u: float = flt_lambda_u
        self.int_epochs_freeze: int = int_epochs_freeze
        self.bl_dropout: bool = bl_dropout

        self.backbone, self.int_backbone_dim = ClfInitialisers.backbone(str_backbone)
        self.fc: nn.Module = ClfInitialisers.dropout(bl_dropout=self.bl_dropout, int_in_features=self.int_backbone_dim, int_out_features=int_classes)
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        self.dct_metrics: nn.ModuleDict = ClfInitialisers.metrics(int_classes=self.int_classes)
        self.dct_metrics_binary: nn.ModuleDict = ClfInitialisers.metrics_binary()
        
        if self.int_epochs_freeze > 0: # freeze backbone initially if needed
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()

        # Per-sample memory of the most recent confident pseudo-label (-1 == unassigned), used to
        # estimate the per-class learning effect sigma_t(c) that drives the curriculum thresholds.
        self.register_buffer("selected_label", torch.full((max(int_unlabeled, 1),), -1, dtype=torch.long))

    def forward(self, tsr_images: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        tsr_features = self.backbone(tsr_images)
        return self.fc(tsr_features.flatten(start_dim=1))

    def _should_train_backbone(self) -> bool:
        return self.current_epoch >= self.int_epochs_freeze

    def _labels_to_class_index(self, tsr_labels: torch.Tensor) -> torch.Tensor:
        """Map raw dataset labels to class indices in [0, int_classes) for cross-entropy training."""
        if self.int_classes <= 2:
            return (tsr_labels.view(-1) > 0).long()
        return tsr_labels.view(-1).long()

    def _class_thresholds(self) -> torch.Tensor:
        """
        Computes the current per-class FlexMatch thresholds from the curriculum memory buffer.
        Uses the "convex" curriculum mapping from the FlexMatch paper: classes with fewer confident pseudo-labels so far 
        (relative to the best-learned class) get a lower threshold, converging to flt_threshold as they become well learned.
        Args:
            self.selected_label: torch.Tensor, per-sample memory of the most recent confident pseudo-label (-1 == unassigned).
        Returns:
            torch.Tensor: Per-class thresholds of shape [int_classes].
        """
        tsr_assigned: torch.Tensor = self.selected_label[self.selected_label != -1]
        tsr_sigma: torch.Tensor = torch.bincount(tsr_assigned, minlength=self.int_classes).float()
        flt_max_sigma: torch.Tensor = tsr_sigma.max()
        if flt_max_sigma <= 0:
            return torch.zeros(self.int_classes, device=self.device)
        tsr_beta: torch.Tensor = tsr_sigma / flt_max_sigma
        return (tsr_beta / (2 - tsr_beta)) * self.flt_threshold
    
    def training_step(self, batch: dict[str, tuple], batch_idx: int) -> torch.Tensor:
        tsr_images_lb, tsr_labels_lb = batch["labeled"]
        tsr_idx_ulb, tsr_images_weak_ulb, tsr_images_strong_ulb = batch["unlabeled"]
        tsr_labels_lb = self._labels_to_class_index(tsr_labels_lb)

        tsr_logits_lb: torch.Tensor = self(tsr_images_lb)
        loss_lb: torch.Tensor = self.loss_fn(tsr_logits_lb, tsr_labels_lb)

        with torch.no_grad():
            tsr_probs_weak: torch.Tensor = torch.softmax(self(tsr_images_weak_ulb), dim=1)
            tsr_max_probs, tsr_pseudo_labels = torch.max(tsr_probs_weak, dim=1)

        tsr_sample_thresholds: torch.Tensor = self._class_thresholds()[tsr_pseudo_labels]
        tsr_mask: torch.Tensor = tsr_max_probs.ge(tsr_sample_thresholds)
        if tsr_mask.any():
            self.selected_label[tsr_idx_ulb[tsr_mask]] = tsr_pseudo_labels[tsr_mask]

        tsr_logits_strong: torch.Tensor = self(tsr_images_strong_ulb)
        if tsr_mask.any():
            loss_ulb: torch.Tensor = F.cross_entropy(tsr_logits_strong[tsr_mask], tsr_pseudo_labels[tsr_mask])
        else:
            loss_ulb: torch.Tensor = torch.zeros((), device=self.device)

        tsr_loss: torch.Tensor = loss_lb + self.flt_lambda_u * loss_ulb

        metrics = cast(MetricCollection, self.dct_metrics["train_split"])
        metrics.update(torch.argmax(tsr_logits_lb, dim=1), tsr_labels_lb)
        int_batch_size: int = tsr_images_lb.size(0)
        self.log("train_loss", tsr_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=int_batch_size, sync_dist=True)
        self.log("train_loss_labeled", loss_lb, on_step=False, on_epoch=True, batch_size=int_batch_size, sync_dist=True)
        self.log("train_loss_unlabeled", loss_ulb, on_step=False, on_epoch=True, batch_size=int_batch_size, sync_dist=True)
        self.log("train_mask_ratio", tsr_mask.float().mean(), on_step=False, on_epoch=True, batch_size=int_batch_size, sync_dist=True)
        self.log("train_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True, batch_size=int_batch_size, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=int_batch_size, sync_dist=True)
        return tsr_loss

    def _shared_prediction_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Shared logic for generating predictions from a batch."""
        tsr_images, tsr_labels = batch
        tsr_labels = self._labels_to_class_index(tsr_labels)
        tsr_logits: torch.Tensor = self(tsr_images)
        tsr_preds_multi: torch.Tensor = torch.argmax(tsr_logits, dim=1)
        
        if self.int_classes <= 2:
            tsr_preds_binary: torch.Tensor = tsr_preds_multi
            tsr_labels_binary: torch.Tensor = tsr_labels
        else:
            tsr_preds_binary: torch.Tensor = (tsr_preds_multi > 0).long()
            tsr_labels_binary: torch.Tensor = (tsr_labels > 0).long()
        
        return {
            "tsr_logits": tsr_logits,
            "tsr_preds_multi": tsr_preds_multi,
            "tsr_preds_binary": tsr_preds_binary,
            "tsr_labels_multi": tsr_labels,
            "tsr_labels_binary": tsr_labels_binary,
        }

    def _eval_step(self, batch: tuple[torch.Tensor, torch.Tensor], str_split: str) -> torch.Tensor:
        """Shared evaluation logic for validation and test steps (labelled data only)."""
        dct_preds = self._shared_prediction_step(batch)
        tsr_loss: torch.Tensor = self.loss_fn(dct_preds["tsr_logits"], dct_preds["tsr_labels_multi"])
        self.dct_metrics_binary[str_split].update(dct_preds["tsr_preds_binary"], dct_preds["tsr_labels_binary"])
        self.dct_metrics[str_split].update(dct_preds["tsr_preds_multi"], dct_preds["tsr_labels_multi"])
        return tsr_loss

    def _log_metrics(self, str_split: str, batch: tuple[torch.Tensor, torch.Tensor], tsr_loss: torch.Tensor) -> None:
        """Log metrics for the given split (val or test)."""
        str_split_full: str = f"{str_split}_split"
        metrics = cast(MetricCollection, self.dct_metrics[str_split_full])
        self.log(f"{str_split}_loss", tsr_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log(f"{str_split}_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._eval_step(batch, "val_split")
        self._log_metrics(str_split="val", batch=batch, tsr_loss=tsr_loss)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._eval_step(batch, "test_split")
        self._log_metrics(str_split="test", batch=batch, tsr_loss=tsr_loss)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Generate predictions for inference on labelled data (standard tuple format)."""
        return self._shared_prediction_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([
            {"params": self.fc.parameters(), "lr": self.flt_lr, "weight_decay": self.flt_lr * 10},
            {"params": self.backbone.parameters(), "lr": self.flt_lr * 0.1, "weight_decay": self.flt_lr}, 
        ])

    def _shared_epoch_start(self, str_split: str) -> None:
        self.dct_metrics[f"{str_split}_split"].reset()

    def on_train_epoch_start(self) -> None:
        self._shared_epoch_start("train")
        if self._should_train_backbone():
            self.backbone.train()
            for param in self.backbone.parameters(): # unfreeze backbone parameters
                param.requires_grad = True
        else:
            self.backbone.eval()
            for param in self.backbone.parameters(): # freeze backbone parameters
                param.requires_grad = False

    def on_validation_epoch_start(self) -> None:
        self._shared_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self._shared_epoch_start("test")


class LSTMClassifier(L.LightningModule):
    """
    LSTM-based temporal classification model that operates on top of a pretrained ClassificationBase backbone.
    Processes sequences of frame features extracted from a ClassificationBase model to perform temporal classification.
    The backbone is frozen during training; only the LSTM and classifier head are trained.
    """
    def __init__(
            self,
            pretrained_model: L.LightningModule,
            int_lstm_hidden_dim: int = 256,
            int_lstm_num_layers: int = 2,
            flt_lstm_dropout: float = 0.3,
            flt_lr: float = 1e-3,
            bl_bidirectional: bool = True,
        ):
        """
        Args:
            pretrained_model: A pretrained ClassificationBase model to extract frame features.
            int_lstm_hidden_dim: Number of hidden units in the LSTM (default: 256).
            int_lstm_num_layers: Number of LSTM layers (default: 2).
            flt_lstm_dropout: Dropout rate for the LSTM (default: 0.3).
            flt_lr: Learning rate for the optimizer (default: 1e-3).
            bl_bidirectional: Whether to use a bidirectional LSTM (default: True).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])
        self.pretrained_model = pretrained_model
        self.int_classes = pretrained_model.int_classes
        self.int_backbone_dim = pretrained_model.int_backbone_dim
        self.int_lstm_hidden_dim = int_lstm_hidden_dim
        self.flt_lr = flt_lr
        self.bl_bidirectional = bl_bidirectional
        
        for param in self.pretrained_model.backbone.parameters():
            param.requires_grad = False # freeze the pretrained backbone
        self.pretrained_model.backbone.eval()
        
        self.lstm = nn.LSTM(
            input_size=self.int_backbone_dim,
            hidden_size=int_lstm_hidden_dim,
            num_layers=int_lstm_num_layers,
            batch_first=True,
            dropout=flt_lstm_dropout if int_lstm_num_layers > 1 else 0,
            bidirectional=bl_bidirectional,
        )
        self.fc = nn.Linear(
            in_features=int_lstm_hidden_dim * (2 if bl_bidirectional else 1), 
            out_features=1 if self.int_classes <= 2 else self.int_classes
        )
        
        self.loss_fn: nn.Module = ClfInitialisers.loss(int_classes=self.int_classes)
        self.dct_metrics_binary: nn.ModuleDict = ClfInitialisers.metrics_binary()
        self.dct_metrics: nn.ModuleDict = ClfInitialisers.metrics(int_classes=self.int_classes)

    def forward(self, tsr_clips: torch.Tensor) -> torch.Tensor:
        batch_size, int_clip, _, height, width = tsr_clips.shape
        tsr_flat = tsr_clips.view(batch_size * int_clip, 3, height, width)
        
        with torch.no_grad():
            tsr_features_flat = self.pretrained_model.backbone(tsr_flat)
        
        tsr_features = tsr_features_flat.view(batch_size, int_clip, self.int_backbone_dim)
        _tsr_lstm_out, (tsr_h_n, _tsr_c_n) = self.lstm(tsr_features)
        
        if self.bl_bidirectional:
            tsr_lstm_final = torch.cat([tsr_h_n[-2], tsr_h_n[-1]], dim=1) # [batch_size, 2*hidden_dim]
        else:
            tsr_lstm_final = tsr_h_n[-1] # [batch_size, hidden_dim]
        
        return self.fc(tsr_lstm_final)

    def _shared_prediction_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Shared logic for generating predictions from a batch of temporal clips.
        Args:
            batch: A tuple containing temporal clip batches and their labels.
        Returns:
            A dictionary containing logits, labels, and predictions for loss and metric computation.
        """
        tsr_clips, tsr_labels = batch
        tsr_logits: torch.Tensor = self(tsr_clips)

        if self.int_classes <= 2:
            tsr_preds_loss: torch.Tensor = tsr_logits.view(-1).float()
            tsr_labels_loss: torch.Tensor = (tsr_labels.view(-1) > 0).float()
            tsr_preds_multi: torch.Tensor = (torch.sigmoid(tsr_preds_loss) > 0.5).long()
            tsr_labels_multi: torch.Tensor = (tsr_labels.view(-1) > 0).long()
            tsr_preds_binary: torch.Tensor = tsr_preds_multi
            tsr_labels_binary: torch.Tensor = tsr_labels_multi
        else:
            tsr_preds_loss: torch.Tensor = tsr_logits.float()
            tsr_labels_loss: torch.Tensor = tsr_labels.view(-1).long()
            tsr_preds_multi: torch.Tensor = torch.argmax(tsr_logits, dim=1)
            tsr_labels_multi: torch.Tensor = tsr_labels.view(-1).long()
            tsr_preds_binary: torch.Tensor = (tsr_preds_multi > 0).long()
            tsr_labels_binary: torch.Tensor = (tsr_labels_multi > 0).long()

        return {
            "tsr_preds_loss": tsr_preds_loss,
            "tsr_labels_loss": tsr_labels_loss,
            "tsr_preds_multi": tsr_preds_multi,
            "tsr_labels_multi": tsr_labels_multi,
            "tsr_preds_binary": tsr_preds_binary,
            "tsr_labels_binary": tsr_labels_binary,
        }
    
    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], str_split: str) -> torch.Tensor:
        """
        Shared logic for training, validation, and testing steps.
        Args:
            batch: A tuple containing temporal clip batches and their labels.
            str_split: The stage ("train_split", "val_split", or "test_split") for metric tracking.
        Returns:
            tsr_loss: The computed loss for the batch.
        """
        dct_output: dict[str, torch.Tensor] = self._shared_prediction_step(batch=batch)

        tsr_loss: torch.Tensor = self.loss_fn(dct_output["tsr_preds_loss"], dct_output["tsr_labels_loss"])
        self.dct_metrics[str_split].update(dct_output["tsr_preds_multi"], dct_output["tsr_labels_multi"])
        self.dct_metrics_binary[str_split].update(dct_output["tsr_preds_binary"], dct_output["tsr_labels_binary"])
        return tsr_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW([
            {"params": self.fc.parameters(), "lr": self.flt_lr, "weight_decay": self.flt_lr * 10},
            {"params": self.lstm.parameters(), "lr": self.flt_lr * 0.1, "weight_decay": self.flt_lr},
        ])

    def _log_metrics(self, str_split: str, batch: tuple[torch.Tensor, torch.Tensor], tsr_loss: torch.Tensor) -> None:
        """Log metrics for the given split (train, val, or test)."""
        str_split_full: str = f"{str_split}_split"
        metrics = cast(MetricCollection, self.dct_metrics[str_split_full])
        clf_metrics_binary = cast(MetricCollection, self.dct_metrics_binary[str_split_full])
        self.log(f"{str_split}_loss", tsr_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log(f"{str_split}_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log(f"{str_split}_f1_binary", clf_metrics_binary["f1_binary"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)
        self.log_dict(clf_metrics_binary, on_step=False, on_epoch=True, batch_size=batch[0].size(0), sync_dist=True)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        tsr_loss = self._shared_step(batch, "train_split")
        self._log_metrics(str_split="train", batch=batch, tsr_loss=tsr_loss)
        return tsr_loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._shared_step(batch, "val_split")
        self._log_metrics(str_split="val", batch=batch, tsr_loss=tsr_loss)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tsr_loss = self._shared_step(batch, "test_split")
        self._log_metrics(str_split="test", batch=batch, tsr_loss=tsr_loss)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self._shared_prediction_step(batch=batch)

    def _shared_epoch_start(self, str_split: str) -> None:
        """Reset metrics at the start of each epoch for the specified split."""
        self.dct_metrics[f"{str_split}_split"].reset()
        self.dct_metrics_binary[f"{str_split}_split"].reset()

    def on_train_epoch_start(self) -> None:
        self._shared_epoch_start("train")

    def on_validation_epoch_start(self) -> None:
        self._shared_epoch_start("val")

    def on_test_epoch_start(self) -> None:
        self._shared_epoch_start("test")


def moco_backbone(checkpoint_path: str) -> tuple[nn.Module, int]:
    backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "classy_state_dict" in state_dict:
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
    target_layer = "layer1.0.conv1.weight"
    prefix = None
    
    for key in state_dict.keys():
        if target_layer in key:
            prefix = key.split(target_layer)[0]
            break
            
    if prefix is None:
        logger.critical("CRITICAL ERROR: Could not find ResNet layers in this checkpoint.")
        logger.critical("Here are the first 10 keys found instead: %s", list(state_dict.keys())[:10])
        raise ValueError("Invalid checkpoint structure.")
    logger.info(f"Success: Auto-detected layer prefix '{prefix}'")

    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix) and "encoder_k" not in key and "queue" not in key:
            clean_key = key[len(prefix):]
            clean_state_dict[clean_key] = value

    missing_keys, unexpected_keys = backbone.load_state_dict(clean_state_dict, strict=False)
    
    logger.info(f"Missing keys (Expected classification head): {missing_keys}")
    logger.info(f"Unexpected keys (Ignored): {unexpected_keys}")
    return backbone, 2048