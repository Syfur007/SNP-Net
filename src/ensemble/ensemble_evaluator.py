"""Ensemble evaluator for computing metrics and comparisons.

This module provides utilities for evaluating ensemble models and comparing
them against individual base models.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    ConfusionMatrix,
)

from src.ensemble.voting_ensemble import WeightedSoftVotingEnsemble

log = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Evaluator for ensemble models with comprehensive metrics.
    
    This class handles:
    - Computing metrics for ensemble and individual models
    - Generating comparison tables
    - Analyzing variance reduction
    - Saving results to disk
    """
    
    def __init__(
        self,
        ensemble: WeightedSoftVotingEnsemble,
        task: str = "binary",
        num_classes: int = 2,
    ):
        """Initialize the ensemble evaluator.
        
        Args:
            ensemble: The weighted soft-voting ensemble to evaluate.
            task: Task type ('binary' or 'multiclass').
            num_classes: Number of classes.
        """
        self.ensemble = ensemble
        self.task = task
        self.num_classes = num_classes
        
        # Initialize metrics for ensemble
        self.metrics = self._create_metrics()
        
        # Storage for results
        self.results = {
            "ensemble": {},
            "individual": {},
            "comparison": None,
        }
    
    def _create_metrics(self) -> Dict[str, Any]:
        """Create torchmetrics objects.
        
        Returns:
            Dictionary of metric objects.
        """
        return {
            "accuracy": Accuracy(task=self.task, num_classes=self.num_classes),
            "precision": Precision(task=self.task, num_classes=self.num_classes),
            "recall": Recall(task=self.task, num_classes=self.num_classes),
            "f1": F1Score(task=self.task, num_classes=self.num_classes),
            "auroc": AUROC(task=self.task, num_classes=self.num_classes),
            "confusion_matrix": ConfusionMatrix(task=self.task, num_classes=self.num_classes),
        }
    
    def _reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the ensemble and individual models on test data.
        
        Args:
            dataloader: DataLoader for test data.
            save_predictions: Whether to save predictions and probabilities.
            
        Returns:
            Dictionary containing all evaluation results.
        """
        log.info("Starting ensemble evaluation...")
        
        # Storage for all predictions and targets
        all_ensemble_preds = []
        all_ensemble_probs = []
        all_individual_probs = {name: [] for name in self.ensemble.model_names}
        all_targets = []
        
        # Iterate over test data
        self.ensemble.models[0].eval()  # Ensure eval mode
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Extract data and targets
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x = batch[0]
                        y = batch[-1]
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                
                # Get ensemble predictions
                ensemble_preds, ensemble_probs, individual_probs = self.ensemble.predict(x)
                
                # Store predictions
                all_ensemble_preds.append(ensemble_preds)
                all_ensemble_probs.append(ensemble_probs)
                all_targets.append(y)
                
                # Store individual model probabilities
                for i, name in enumerate(self.ensemble.model_names):
                    all_individual_probs[name].append(individual_probs[i])
        
        # Concatenate all batches
        all_ensemble_preds = torch.cat(all_ensemble_preds, dim=0)
        all_ensemble_probs = torch.cat(all_ensemble_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        for name in self.ensemble.model_names:
            all_individual_probs[name] = torch.cat(all_individual_probs[name], dim=0)
        
        log.info(f"Processed {len(all_targets)} test samples.")
        
        # Compute ensemble metrics
        log.info("Computing ensemble metrics...")
        ensemble_metrics = self._compute_metrics(
            preds=all_ensemble_preds,
            probs=all_ensemble_probs,
            targets=all_targets,
        )
        self.results["ensemble"] = ensemble_metrics
        
        # Compute individual model metrics
        log.info("Computing individual model metrics...")
        individual_metrics = {}
        for name in self.ensemble.model_names:
            probs = all_individual_probs[name]
            preds = torch.argmax(probs, dim=1)
            
            metrics = self._compute_metrics(
                preds=preds,
                probs=probs,
                targets=all_targets,
            )
            individual_metrics[name] = metrics
        
        self.results["individual"] = individual_metrics
        
        # Generate comparison table
        log.info("Generating comparison table...")
        comparison_df = self._create_comparison_table()
        self.results["comparison"] = comparison_df
        
        # Compute variance analysis
        log.info("Computing variance analysis...")
        variance_analysis = self._compute_variance_analysis(all_individual_probs, all_targets)
        self.results["variance_analysis"] = variance_analysis
        
        # Save predictions if requested
        if save_predictions:
            self.results["predictions"] = {
                "ensemble_preds": all_ensemble_preds.numpy(),
                "ensemble_probs": all_ensemble_probs.numpy(),
                "individual_probs": {
                    name: probs.numpy() for name, probs in all_individual_probs.items()
                },
                "targets": all_targets.numpy(),
            }
        
        log.info("Evaluation complete!")
        
        return self.results
    
    def _compute_metrics(
        self,
        preds: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all metrics for given predictions.
        
        Args:
            preds: Predicted class labels.
            probs: Predicted probabilities.
            targets: Ground truth labels.
            
        Returns:
            Dictionary of computed metrics.
        """
        # Reset metrics
        self._reset_metrics()
        
        # For binary classification, extract positive class probability for AUROC
        if self.task == "binary" and self.num_classes == 2:
            probs_for_auroc = probs[:, 1]
        else:
            probs_for_auroc = probs
        
        # Update metrics
        self.metrics["accuracy"].update(preds, targets)
        self.metrics["precision"].update(preds, targets)
        self.metrics["recall"].update(preds, targets)
        self.metrics["f1"].update(preds, targets)
        self.metrics["auroc"].update(probs_for_auroc, targets)
        self.metrics["confusion_matrix"].update(preds, targets)
        
        # Compute final values
        results = {
            "accuracy": self.metrics["accuracy"].compute().item(),
            "precision": self.metrics["precision"].compute().item(),
            "recall": self.metrics["recall"].compute().item(),
            "f1": self.metrics["f1"].compute().item(),
            "auroc": self.metrics["auroc"].compute().item(),
            "confusion_matrix": self.metrics["confusion_matrix"].compute().numpy(),
        }
        
        return results
    
    def _create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of ensemble vs individual models.
        
        Returns:
            DataFrame with metrics for all models.
        """
        rows = []
        
        # Add individual model metrics
        for name in self.ensemble.model_names:
            metrics = self.results["individual"][name]
            rows.append({
                "Model": name,
                "Type": "Individual",
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1"],
                "ROC-AUC": metrics["auroc"],
            })
        
        # Add ensemble metrics
        ensemble_metrics = self.results["ensemble"]
        rows.append({
            "Model": "Ensemble",
            "Type": "Ensemble",
            "Accuracy": ensemble_metrics["accuracy"],
            "Precision": ensemble_metrics["precision"],
            "Recall": ensemble_metrics["recall"],
            "F1-Score": ensemble_metrics["f1"],
            "ROC-AUC": ensemble_metrics["auroc"],
        })
        
        df = pd.DataFrame(rows)
        
        # Add improvement column (ensemble vs best individual)
        individual_accs = [self.results["individual"][name]["accuracy"] 
                          for name in self.ensemble.model_names]
        best_individual_acc = max(individual_accs)
        
        df["Improvement (%)"] = 0.0
        df.loc[df["Model"] == "Ensemble", "Improvement (%)"] = (
            (ensemble_metrics["accuracy"] - best_individual_acc) / best_individual_acc * 100
        )
        
        return df
    
    def _compute_variance_analysis(
        self,
        all_individual_probs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute variance analysis across models.
        
        Args:
            all_individual_probs: Dictionary of individual model probabilities.
            targets: Ground truth labels.
            
        Returns:
            Dictionary with variance analysis results.
        """
        # Extract accuracies
        accuracies = []
        for name in self.ensemble.model_names:
            probs = all_individual_probs[name]
            preds = torch.argmax(probs, dim=1)
            acc = (preds == targets).float().mean().item()
            accuracies.append(acc)
        
        accuracies = np.array(accuracies)
        
        # Compute statistics
        analysis = {
            "mean_accuracy": accuracies.mean(),
            "std_accuracy": accuracies.std(),
            "min_accuracy": accuracies.min(),
            "max_accuracy": accuracies.max(),
            "ensemble_accuracy": self.results["ensemble"]["accuracy"],
        }
        
        # Variance reduction relative to best model (e.g., Bi-LSTM)
        bilstm_acc = accuracies[0] if len(accuracies) > 0 else 0.0
        analysis["variance_reduction_vs_bilstm"] = (
            (bilstm_acc - analysis["std_accuracy"]) / bilstm_acc * 100
            if bilstm_acc > 0 else 0.0
        )
        
        # Compute pairwise prediction agreement
        predictions = []
        for name in self.ensemble.model_names:
            probs = all_individual_probs[name]
            preds = torch.argmax(probs, dim=1)
            predictions.append(preds.numpy())
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Agreement matrix
        n_models = len(self.ensemble.model_names)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                agreement = (predictions[i] == predictions[j]).mean()
                agreement_matrix[i, j] = agreement
        
        analysis["agreement_matrix"] = agreement_matrix
        analysis["mean_agreement"] = agreement_matrix[np.triu_indices(n_models, k=1)].mean()
        
        return analysis
    
    def save_results(self, output_dir: Path):
        """Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Saving results to {output_dir}...")
        
        # Save comparison table
        comparison_path = output_dir / "model_comparison.csv"
        self.results["comparison"].to_csv(comparison_path, index=False)
        log.info(f"  ✓ Saved comparison table to {comparison_path}")
        
        # Save ensemble metrics
        ensemble_metrics_path = output_dir / "ensemble_metrics.txt"
        with open(ensemble_metrics_path, "w") as f:
            f.write("Ensemble Metrics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in self.results["ensemble"].items():
                if key != "confusion_matrix":
                    f.write(f"{key}: {value:.4f}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(str(self.results["ensemble"]["confusion_matrix"]))
        
        log.info(f"  ✓ Saved ensemble metrics to {ensemble_metrics_path}")
        
        # Save variance analysis
        variance_path = output_dir / "variance_analysis.txt"
        with open(variance_path, "w") as f:
            f.write("Variance Analysis\n")
            f.write("=" * 50 + "\n\n")
            for key, value in self.results["variance_analysis"].items():
                if key not in ["agreement_matrix"]:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n\nAgreement Matrix:\n")
            f.write(str(self.results["variance_analysis"]["agreement_matrix"]))
        
        log.info(f"  ✓ Saved variance analysis to {variance_path}")
        
        # Save predictions if available
        if "predictions" in self.results:
            predictions_path = output_dir / "predictions.npz"
            np.savez(
                predictions_path,
                ensemble_preds=self.results["predictions"]["ensemble_preds"],
                ensemble_probs=self.results["predictions"]["ensemble_probs"],
                targets=self.results["predictions"]["targets"],
                **{f"{name}_probs": probs 
                   for name, probs in self.results["predictions"]["individual_probs"].items()}
            )
            log.info(f"  ✓ Saved predictions to {predictions_path}")
        
        # Save confusion matrix
        cm_path = output_dir / "confusion_matrix.csv"
        cm_df = pd.DataFrame(
            self.results["ensemble"]["confusion_matrix"],
            index=[f"True_{i}" for i in range(self.num_classes)],
            columns=[f"Pred_{i}" for i in range(self.num_classes)],
        )
        cm_df.to_csv(cm_path)
        log.info(f"  ✓ Saved confusion matrix to {cm_path}")
        
        log.info("All results saved successfully!")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION SUMMARY")
        print("=" * 70 + "\n")
        
        # Print comparison table
        print("Model Comparison:")
        print("-" * 70)
        print(self.results["comparison"].to_string(index=False))
        print()
        
        # Print variance analysis
        print("\nVariance Analysis:")
        print("-" * 70)
        variance = self.results["variance_analysis"]
        print(f"Mean Accuracy: {variance['mean_accuracy']:.4f}")
        print(f"Std Accuracy: {variance['std_accuracy']:.4f}")
        print(f"Min Accuracy: {variance['min_accuracy']:.4f}")
        print(f"Max Accuracy: {variance['max_accuracy']:.4f}")
        print(f"Ensemble Accuracy: {variance['ensemble_accuracy']:.4f}")
        print(f"Mean Agreement: {variance['mean_agreement']:.4f}")
        
        print("\n" + "=" * 70 + "\n")
