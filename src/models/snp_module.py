from typing import Any, Dict, Tuple

import torch
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


class SNPLitModule(LightningModule):
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
        """Initialize a `SNPLitModule`.

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
        self.val_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auroc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

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

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return loss, preds, probs, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, probs, targets = self.model_step(batch)

        # For binary classification, extract probability of positive class
        if self.hparams.num_classes == 2:
            probs_for_auroc = probs[:, 1]  # Probability of class 1
        else:
            probs_for_auroc = probs

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        self.train_auroc(probs_for_auroc, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs, targets = self.model_step(batch)

        # For binary classification, extract probability of positive class
        if self.hparams.num_classes == 2:
            probs_for_auroc = probs[:, 1]  # Probability of class 1
        else:
            probs_for_auroc = probs

        # update and log metrics
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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of SNP features and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, probs, targets = self.model_step(batch)

        # For binary classification, extract probability of positive class
        if self.hparams.num_classes == 2:
            probs_for_auroc = probs[:, 1]  # Probability of class 1
        else:
            probs_for_auroc = probs

        # update and log metrics
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

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # Build the model with a sample batch to ensure parameters exist before optimizer setup
        if stage == "fit" and hasattr(self.net, 'model') and self.net.model is None:
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
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SNPLitModule(None, None, None, None)
