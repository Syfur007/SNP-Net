# Model Summary (Default Configs)

This table summarizes the default model configurations (from configs/model) and shared defaults from data/trainer/callbacks.

| Model | Architecture | Loss function | Learning rate | Weight decay | LR scheduler | Batch size | Epochs | Early stop |
|---|---|---|---|---|---|---|---|---|
| autoencoder | src.models.components.autoencoder_net.AutoencoderNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| bilstm | src.models.components.bilstm_net.BiLSTMNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| deepplantcre | src.models.components.deepplantcre_net.DeepPlantCRENet | `torch.nn.CrossEntropyLoss` | 0.0001 | 0.0 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| dense | src.models.components.dense_net.DenseNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| dpcformer | src.models.components.dpcformer_net.DPCformerNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| gru | src.models.components.gru_net.GRUNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| lstm | src.models.components.lstm_net.LSTMNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| stacked_lstm | src.models.components.stacked_lstm_net.StackedLSTMNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| transformer_cnn | src.models.components.transformer_cnn_net.TransformerCNNNet | `torch.nn.CrossEntropyLoss` | 0.0005 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| vae | src.models.components.vae_net.VAENet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
| wheatgp | src.models.components.wheatgp_net.WheatGPNet | `torch.nn.CrossEntropyLoss` | 0.001 | 0.0001 | reduce_lr_on_plateau | 32 | min=1, max=100 | monitor=val/acc, patience=5, mode=max |
