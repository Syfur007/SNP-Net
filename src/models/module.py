from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)


class LitModule(LightningModule):
    """LightningModule for SNP (Single Nucleotide Polymorphism) classification.

    This module is designed for classification tasks using SNP data.
    It supports both binary and multi-class classification.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 2,
    ) -> None:
        """Initialize a `LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with torch.compile.
        :param num_classes: Number of classes for classification.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Determine task type
        task = "binary" if num_classes == 2 else "multiclass"

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.test_acc = Accuracy(task=task, num_classes=num_classes)

        # Additional metrics for comprehensive evaluation
        self.train_precision = Precision(task=task, num_classes=num_classes)
        self.val_precision = Precision(task=task, num_classes=num_classes)
        self.test_precision = Precision(task=task, num_classes=num_classes)

        self.train_recall = Recall(task=task, num_classes=num_classes)
        self.val_recall = Recall(task=task, num_classes=num_classes)
        self.test_recall = Recall(task=task, num_classes=num_classes)

        self.train_f1 = F1Score(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task=task, num_classes=num_classes)
        self.test_f1 = F1Score(task=task, num_classes=num_classes)

        self.train_auroc = AUROC(task=task, num_classes=num_classes)
        self.val_auroc = AUROC(task=task, num_classes=num_classes)
        self.test_auroc = AUROC(task=task, num_classes=num_classes)

        # Confusion matrix for detailed analysis
        self.train_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.val_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Class-wise loss tracking
        self.train_loss_per_class = torch.nn.ModuleList([MeanMetric() for _ in range(num_classes)])
        self.val_loss_per_class = torch.nn.ModuleList([MeanMetric() for _ in range(num_classes)])
        self.test_loss_per_class = torch.nn.ModuleList([MeanMetric() for _ in range(num_classes)])

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auroc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        # Containers for test outputs
        self._test_probs: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

    def _compute_sensitivity_specificity(
        self, confusion_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Compute sensitivity and specificity from a confusion matrix.

        :param confusion_matrix: Confusion matrix tensor of shape (C, C).
        :return: Tuple of (sensitivity_macro, specificity_macro, sensitivity_per_class, specificity_per_class).
        """
        cm = confusion_matrix.float()
        num_classes = cm.size(0)

        sensitivity_per_class: List[torch.Tensor] = []
        specificity_per_class: List[torch.Tensor] = []

        for class_idx in range(num_classes):
            tp = cm[class_idx, class_idx]
            fn = cm[class_idx, :].sum() - tp
            fp = cm[:, class_idx].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=cm.device)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0, device=cm.device)

            sensitivity_per_class.append(sensitivity)
            specificity_per_class.append(specificity)

        sensitivity_macro = torch.stack(sensitivity_per_class).mean()
        specificity_macro = torch.stack(specificity_per_class).mean()

        return sensitivity_macro, specificity_macro, sensitivity_per_class, specificity_per_class

    def _log_confusion_matrix_image(self, confusion_matrix: torch.Tensor, stage: str) -> None:
        """Log confusion matrix as an image to all available loggers.

        :param confusion_matrix: Confusion matrix tensor of shape (C, C).
        :param stage: Stage name, e.g., "test".
        """
        if not self.trainer or not self.trainer.loggers:
            return

        if not self.trainer.is_global_zero:
            return

        cm = confusion_matrix.detach().cpu().numpy()
        num_classes = cm.shape[0]
        class_labels = [str(i) for i in range(num_classes)]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_labels,
            yticklabels=class_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    f"{int(cm[i, j])}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        fig.tight_layout()

        tag = f"{stage}/confusion_matrix"
        for logger in self.trainer.loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue

            # TensorBoard
            if hasattr(experiment, "add_figure"):
                experiment.add_figure(tag, fig, global_step=self.global_step)
                continue

            # Weights & Biases
            if experiment.__class__.__module__.startswith("wandb"):
                try:
                    import wandb

                    experiment.log({tag: wandb.Image(fig)}, step=self.global_step)
                except Exception:
                    pass
                continue

            # MLflow / Comet / Neptune or other loggers with figure support
            if hasattr(experiment, "log_figure"):
                try:
                    experiment.log_figure(figure=fig, figure_name=f"{tag}.png", step=self.global_step)
                except Exception:
                    try:
                        experiment.log_figure(figure=fig, figure_name=f"{tag}.png")
                    except Exception:
                        pass
                continue

            if hasattr(experiment, "log_image"):
                try:
                    experiment.log_image(name=tag, image=fig, step=self.global_step)
                except Exception:
                    try:
                        experiment.log_image(tag, fig)
                    except Exception:
                        pass

        plt.close(fig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of SNP features.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_acc_best.reset()
        self.val_auroc_best.reset()
        self.val_f1_best.reset()
        self.train_confusion_matrix.reset()
        for metric in self.train_loss_per_class:
            metric.reset()
        for metric in self.val_loss_per_class:
            metric.reset()
        for metric in self.test_loss_per_class:
            metric.reset()

    def on_train_epoch_start(self) -> None:
        self.train_confusion_matrix.reset()
        for metric in self.train_loss_per_class:
            metric.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_confusion_matrix.reset()
        for metric in self.val_loss_per_class:
            metric.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of probabilities.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        return loss, preds, probs, y, logits

    def _get_probs_for_auroc(self, probs: torch.Tensor) -> torch.Tensor:
        """Select probability tensor for AUROC computation based on number of classes."""
        if self.hparams.num_classes == 2:
            return probs[:, 1]
        return probs

    def _update_and_log_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        preds: torch.Tensor,
        probs_for_auroc: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update metric objects and log them for the given stage."""
        if stage == "train":
            self.train_loss(loss)
            self.train_acc(preds, targets)
            self.train_precision(preds, targets)
            self.train_recall(preds, targets)
            self.train_f1(preds, targets)
            self.train_auroc(probs_for_auroc, targets)
            self.train_confusion_matrix(preds, targets)

            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)
            return

        if stage == "val":
            self.val_loss(loss)
            self.val_acc(preds, targets)
            self.val_precision(preds, targets)
            self.val_recall(preds, targets)
            self.val_f1(preds, targets)
            self.val_auroc(probs_for_auroc, targets)
            self.val_confusion_matrix(preds, targets)

            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
            return

        if stage == "test":
            self.test_loss(loss)
            self.test_acc(preds, targets)
            self.test_precision(preds, targets)
            self.test_recall(preds, targets)
            self.test_f1(preds, targets)
            self.test_auroc(probs_for_auroc, targets)
            self.test_confusion_matrix(preds, targets)

            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, probs, targets, logits = self.model_step(batch)

        probs_for_auroc = self._get_probs_for_auroc(probs)
        self._update_and_log_metrics("train", loss, preds, probs_for_auroc, targets)
        self._update_classwise_losses("train", logits, targets)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self._log_classwise_losses("train")
        self._log_confusion_matrix_metrics("train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs, targets, logits = self.model_step(batch)

        probs_for_auroc = self._get_probs_for_auroc(probs)
        self._update_and_log_metrics("val", loss, preds, probs_for_auroc, targets)
        self._update_classwise_losses("val", logits, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        auroc = self.val_auroc.compute()  # get current val auroc
        f1 = self.val_f1.compute()  # get current val f1
        
        self.val_acc_best(acc)  # update best so far val acc
        self.val_auroc_best(auroc)  # update best so far val auroc
        self.val_f1_best(f1)  # update best so far val f1
        
        # log best metrics
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=False)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=False)

        self._log_classwise_losses("val")
        self._log_confusion_matrix_metrics("val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs, targets, logits = self.model_step(batch)

        probs_for_auroc = self._get_probs_for_auroc(probs)
        self._update_and_log_metrics("test", loss, preds, probs_for_auroc, targets)
        self._update_classwise_losses("test", logits, targets)

        # Store outputs for downstream metrics/curves
        self._test_probs.append(probs.detach().cpu())
        self._test_targets.append(targets.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_cm = self.test_confusion_matrix.compute()
        test_sensitivity, test_specificity, test_sensitivity_pc, test_specificity_pc = (
            self._compute_sensitivity_specificity(test_cm)
        )

        self.log("test/sensitivity", test_sensitivity, sync_dist=True, prog_bar=False)
        self.log("test/specificity", test_specificity, sync_dist=True, prog_bar=False)

        if self.hparams.num_classes > 2:
            for idx, value in enumerate(test_sensitivity_pc):
                self.log(f"test/sensitivity_class_{idx}", value, sync_dist=True, prog_bar=False)
            for idx, value in enumerate(test_specificity_pc):
                self.log(f"test/specificity_class_{idx}", value, sync_dist=True, prog_bar=False)

        self._log_confusion_matrix_image(test_cm, stage="test")
        self._log_classwise_losses("test")
        self._log_confusion_matrix_metrics("test")

    def on_test_start(self) -> None:
        self._test_probs = []
        self._test_targets = []
        self.test_confusion_matrix.reset()
        for metric in self.test_loss_per_class:
            metric.reset()

    def _update_classwise_losses(self, stage: str, logits: torch.Tensor, targets: torch.Tensor) -> None:
        losses = F.cross_entropy(logits, targets, reduction="none")
        for class_idx in range(self.hparams.num_classes):
            mask = targets == class_idx
            if mask.any():
                if stage == "train":
                    self.train_loss_per_class[class_idx](losses[mask].mean())
                elif stage == "val":
                    self.val_loss_per_class[class_idx](losses[mask].mean())
                elif stage == "test":
                    self.test_loss_per_class[class_idx](losses[mask].mean())

    def _log_classwise_losses(self, stage: str) -> None:
        if stage == "train":
            metrics = self.train_loss_per_class
        elif stage == "val":
            metrics = self.val_loss_per_class
        else:
            metrics = self.test_loss_per_class

        for idx, metric in enumerate(metrics):
            self.log(f"{stage}/loss_class_{idx}", metric.compute(), sync_dist=True, prog_bar=False)

    def _log_confusion_matrix_metrics(self, stage: str) -> None:
        if stage == "train":
            cm = self.train_confusion_matrix.compute()
        elif stage == "val":
            cm = self.val_confusion_matrix.compute()
        else:
            cm = self.test_confusion_matrix.compute()

        sensitivity_macro, specificity_macro, _, _ = self._compute_sensitivity_specificity(cm)
        balanced_acc = (sensitivity_macro + specificity_macro) / 2.0
        mcc = self._compute_mcc(cm)

        self.log(f"{stage}/balanced_acc", balanced_acc, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/mcc", mcc, sync_dist=True, prog_bar=False)

    def _compute_mcc(self, confusion_matrix: torch.Tensor) -> torch.Tensor:
        cm = confusion_matrix.float()
        t_sum = cm.sum(dim=1)
        p_sum = cm.sum(dim=0)
        n = cm.sum()
        c = torch.trace(cm)
        s = torch.sum(p_sum * t_sum)
        numerator = c * n - s
        denominator = torch.sqrt((n**2 - torch.sum(p_sum**2)) * (n**2 - torch.sum(t_sum**2)))
        if denominator == 0:
            return torch.tensor(0.0, device=cm.device)
        return numerator / denominator

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Build the model with a sample batch to ensure parameters exist before optimizer setup
        # Check if model needs dynamic building (has .model or .encoder attribute that is None)
        needs_building = False
        if hasattr(self.net, 'model') and self.net.model is None:
            needs_building = True
        elif hasattr(self.net, 'encoder') and self.net.encoder is None:
            needs_building = True
        
        if stage == "fit" and needs_building:
            # Get a sample batch from the datamodule to infer input size
            self.trainer.datamodule.setup(stage)
            train_dataloader = self.trainer.datamodule.train_dataloader()
            sample_batch = next(iter(train_dataloader))
            sample_x = sample_batch[0]
            
            # Move to the correct device
            if torch.cuda.is_available():
                sample_x = sample_x.cuda()
            
            # Set to eval mode to avoid BatchNorm issues with small batch
            self.net.eval()
            
            # Do a dummy forward pass to build the model
            with torch.no_grad():
                _ = self.net(sample_x)  # Use full batch
            
            # Set back to train mode
            self.net.train()
            
            # Move model back to CPU if needed (trainer will handle device placement)
            self.net = self.net.cpu()
        
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            
            # Determine scheduler configuration based on scheduler type
            # ReduceLROnPlateau requires metric monitoring, while epoch-based schedulers do not
            scheduler_name = self.hparams.scheduler._target_ if hasattr(self.hparams.scheduler, '_target_') else str(type(scheduler).__name__)
            
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            
            # Add metric monitoring only for metric-based schedulers (ReduceLROnPlateau)
            if "ReduceLROnPlateau" in scheduler_name or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "val/loss"
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitModule(None, None, None, None)
