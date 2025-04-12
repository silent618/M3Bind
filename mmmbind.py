import argparse

from tqdm import tqdm

from Retina_CLIP.flair.pretraining.data.dataloader import get_loader as flair_loader
from Retina_CLIP.flair.pretraining.data.transforms import augmentations_pretraining
from Retina_CLIP.flair.modeling.model import FLAIRModel, ProjectionLayer
from torch.amp import autocast, GradScaler
from Retina_CLIP.local_data.constants import *

import loralib as lora
import torch
import torch.nn.functional as F
from transformer_maskgit import CTViT

from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer

import peft

from torch.nn.parallel import DataParallel

from collections import deque
import random
from CLIP_models import Retina_CLIP, XRay_CLIP, ECG_CLIP, CT_CLIP, Pathology_CLIP

center_device = torch.device('cuda:0')
retina_gpus = ['cuda:1', 'cuda:2']
xray_gpu = 'cuda:3'
pathology_gpu = 'cuda:4'
ct_gpus = ['cuda:5', 'cuda:6']
ecg_gpu = 'cuda:7'

stream_retina = torch.cuda.Stream(retina_gpus[0])
stream_pathology = torch.cuda.Stream(pathology_gpu)
stream_xray = torch.cuda.Stream(xray_gpu)
stream_ct = torch.cuda.Stream(ct_gpus[0])
stream_ecg = torch.cuda.Stream(ecg_gpu)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class TextBind(torch.nn.Module):
    def __init__(self, retina_model=None, pathology_model=None, xray_model=None, ct_model=None, ecg_model=None,
                 iter_num=3000, batch_size=48, lr=2e-5, weight_decay=1e-5,
                 scheduler=True, store_num=300, warmup_iter_num=100):
        super().__init__()
        self.retina_model = retina_model
        self.xray_model = xray_model
        self.pathology_model = pathology_model
        self.ct_model = ct_model
        self.ecg_model = ecg_model
        self.iter_num = iter_num
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.store_num = store_num
        self.warmup_iter_num = warmup_iter_num

        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    def forward(self, retina_images, retina_input_ids, retina_attention_mask, retina_sel,
                pathology_inputs, xray_imgs, xray_caption_ids, xray_attention_mask, xray_token_type_ids,
                ct_text_tokens, ct_video, ecg_imgs, ecg_input_ids, ecg_attention_mask, ecg_token_type_ids,
                center_input_ids, center_attention_mask, center_token_type_id,
                ):
        with torch.cuda.stream(stream_retina):
            retina_loss, _, _, retina_center_embeds = self.retina_model(
                retina_images.to(retina_gpus[0]), retina_input_ids.to(retina_gpus[0]),
                retina_attention_mask.to(retina_gpus[0]),
                retina_sel, center_input_ids.to(retina_gpus[0]), center_attention_mask.to(retina_gpus[0]))

        with torch.cuda.stream(stream_pathology):
            pathology_outputs, _, _ = self.pathology_model(**pathology_inputs.to(pathology_gpu), return_loss=True)
            pathology_center_embeds = self.pathology_model.text_model(center_input_ids.to(pathology_gpu),
                                                                      center_attention_mask.to(pathology_gpu))[
                'pooler_output']
            pathology_loss = pathology_outputs.loss

        with torch.cuda.stream(stream_xray):
            xray_loss, _, _, xray_center_embeds = self.xray_model(
                xray_imgs.to(xray_gpu), xray_caption_ids.to(xray_gpu),
                xray_attention_mask.to(xray_gpu),
                xray_token_type_ids.to(xray_gpu), center_input_ids.to(xray_gpu),
                center_attention_mask.to(xray_gpu), center_token_type_id.to(xray_gpu))

        with torch.cuda.stream(stream_ct):
            ct_loss, _, _ = self.ct_model(ct_text_tokens.to(ct_gpus[0]), ct_video.to(ct_gpus[0]),
                                          return_loss=True)
            ct_center_embeds = self.ct_model.text_transformer(center_input_ids.to(ct_gpus[0]),
                                                              center_attention_mask.to(ct_gpus[0]),
                                                              center_token_type_id.to(ct_gpus[0]))

        with torch.cuda.stream(stream_ecg):
            ecg_loss, _, _ = self.ecg_model(ecg_imgs.to(ecg_gpu), ecg_input_ids.to(ecg_gpu),
                                            ecg_attention_mask.to(ecg_gpu), ecg_token_type_ids.to(ecg_gpu))
            ecg_center_embeds = self.ecg_model.text_encoder(center_input_ids.to(ecg_gpu),
                                                            center_attention_mask.to(ecg_gpu),
                                                            center_token_type_id.to(ecg_gpu))

        retina_center_embeds = retina_center_embeds.to(center_device)
        pathology_center_embeds = pathology_center_embeds.to(center_device)
        xray_center_embeds = xray_center_embeds.to(center_device)
        ct_center_embeds = ct_center_embeds.to(center_device)
        ecg_center_embeds = ecg_center_embeds.to(center_device)
        center_loss = (F.mse_loss(retina_center_embeds, pathology_center_embeds) +
                       F.mse_loss(retina_center_embeds, xray_center_embeds) +
                       F.mse_loss(xray_center_embeds, pathology_center_embeds) +
                       F.mse_loss(ct_center_embeds, retina_center_embeds) +
                       F.mse_loss(ct_center_embeds, xray_center_embeds) +
                       F.mse_loss(ct_center_embeds, pathology_center_embeds) +
                       F.mse_loss(ct_center_embeds, ecg_center_embeds) +
                       F.mse_loss(ecg_center_embeds, xray_center_embeds) +
                       F.mse_loss(ecg_center_embeds, retina_center_embeds) +
                       F.mse_loss(ecg_center_embeds, pathology_center_embeds))

        return retina_loss, pathology_loss, xray_loss, ct_loss, ecg_loss, center_loss
