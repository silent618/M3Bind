import torch
import torch.nn as nn
from transformers import BertModel
from encoder import ECGEncoder


def clip_loss(logits_ecg2text, logits_text2ecg, temperature):
    """对称对比损失"""
    labels = torch.arange(logits_ecg2text.size(0), device=logits_ecg2text.device)
    loss_ecg = nn.functional.cross_entropy(logits_ecg2text, labels)
    loss_text = nn.functional.cross_entropy(logits_text2ecg, labels)
    return (loss_ecg + loss_text) / 2


class ECGCLIP(nn.Module):
    def __init__(self,
                 text_encoder_name='emilyalsentzer/Bio_ClinicalBERT',
                 proj_dim=512,
                 temperature=0.07):
        super(ECGCLIP).__init__()

        self.ecg_encoder = ECGEncoder()

        self.text_encoder = BertModel.from_pretrained(text_encoder_name)

        self.ecg_proj = nn.Linear(self.ecg_encoder.d_model, proj_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / temperature)))

    def forward(self, ecg_input, input_ids, attention_mask, token_type_ids):

        ecg_features = self.ecg_encoder(ecg_input)  # (batch_size, d_model)
        text_features = self.text_encoder(input_ids, attention_mask, token_type_ids)  # (batch_size, hidden_size)

        ecg_emb = self.ecg_proj(ecg_features)  # (batch_size, proj_dim)
        text_emb = self.text_proj(text_features)  # (batch_size, proj_dim)

        ecg_emb = nn.functional.normalize(ecg_emb, p=2, dim=1)
        text_emb = nn.functional.normalize(text_emb, p=2, dim=1)

        loss = clip_loss(ecg_emb, text_emb, self.logit_scale.exp())

        return loss, ecg_emb, text_emb
