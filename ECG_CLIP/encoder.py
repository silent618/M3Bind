import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
    def __init__(self,
                 input_dim=1,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_seq_len=5000):
        super(ECGEncoder).__init__()
        self.d_model = d_model

        self.value_embedding = nn.Linear(input_dim, d_model)

        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.size(0)
        class_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x.unsqueeze(1)], dim=1)  # (batch, 1+seq_len)

        encoded = self.encoder(x.squeeze(-1))
        return encoded[:, 0, :]
