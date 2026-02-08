# Experimental Setup

## Dataset

We evaluated our models on the Autism dataset, containing SNP (Single Nucleotide Polymorphism) data for case-control classification. The dataset was split into training (70%), validation (15%), and test (15%) sets. All features were normalized using z-score normalization to ensure consistent input scaling across models.

## Compared Models

We conducted a comparative analysis of 11 deep learning architectures for autism classification:

1. **Dense** – A multi-layer perceptron (MLP) baseline with fully connected layers
2. **LSTM** – Long Short-Term Memory networks for sequential feature modeling
3. **Bidirectional LSTM (BiLSTM)** – Bidirectional variant capturing both forward and backward dependencies
4. **GRU** – Gated Recurrent Unit networks as a computationally efficient RNN alternative
5. **Stacked LSTM** – Multiple stacked LSTM layers for hierarchical feature representation
6. **Autoencoder** – Unsupervised representation learning with symmetric encoder-decoder architecture
7. **Variational Autoencoder (VAE)** – Probabilistic generative model for latent feature learning
8. **Transformer-CNN (Hybrid)** – Combined architecture integrating transformer's global context with CNN's local pattern extraction
9. **DPCformer** – Deep Pheno Correlation Former hybrid model with multi-head self-attention and CNN modules
10. **DeepPlantCRE** – Transformer-CNN hybrid framework combining convolutional feature extraction with transformer-based attention mechanisms
11. **WheatGP** – Specialized genomic prediction model adapted for SNP data analysis

## Training Configuration

All models were trained using the following standardized hyperparameters:

- **Optimizer:** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **Learning Rate:** 
  - Standard models (Dense, RNN variants): $5 \times 10^{-4}$
  - Transformer-based models (Transformer-CNN, DPCformer, DeepPlantCRE): $1 \times 10^{-4} - 5 \times 10^{-5}$
- **Weight Decay:** $10^{-4}$ to $10^{-5}$ (model-specific)
- **Batch Size:** 32
- **Epochs:** 10–100 (with early stopping)
- **Gradient Clipping:** 0.5–1.0 (varying by architecture)
- **Learning Rate Scheduler:** ReduceLROnPlateau with factor 0.1–0.5, patience 5–10 epochs

## Validation and Early Stopping

Training employed early stopping with patience of 5–10 epochs, monitoring either validation loss or Area Under the ROC Curve (AUROC) as the primary stopping criterion. The top 3 model checkpoints were preserved based on validation accuracy (standard models) or AUROC (hybrid/attention-based models).

## Implementation Details

- **Framework:** PyTorch Lightning with Hydra configuration management
- **Data Loading:** 4 workers with pinned memory for GPU acceleration
- **Feature Input:** 2,000 SNPs for hybrid models (DeepPlantCRE, DPCformer, Transformer-CNN)
- **Reproducibility:** Fixed random seed (12345) for all experiments
- **Logging:** Weights & Biases integration for experiment tracking and metrics visualization
- **Hardware:** CPU/GPU device configuration managed by Lightning

## Evaluation Metrics

Model performance was evaluated on the held-out test set using standard classification metrics: accuracy, precision, recall, F1-score, and Area Under the ROC Curve (AUROC). Results were compared to identify the best-performing architecture for autism case-control classification.
