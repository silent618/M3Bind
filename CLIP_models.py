from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
import torch
from torch import nn
import peft


class CT_CLIP(nn.Module):
    def __init__(self, model, gpu_ids):
        super(CT_CLIP, self).__init__()
        self.model = model

        config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model.text_transformer = peft.LoraModel(self.model.text_transformer, config, "LoRA")
        self.model.visual_transformer = peft.LoraModel(self.model.visual_transformer, config, "LoRA")

        self.model.visual_transformer = nn.DataParallel(self.model.visual_transformer, device_ids=gpu_ids, output_device=gpu_ids[0])

    def forward(self, text, image):
        loss, img_embeds, text_embeds = self.model(text, image)
        return loss, img_embeds, text_embeds


class Retina_CLIP(nn.Module):
    def __init__(self, model, gpu_ids):
        super(Retina_CLIP, self).__init__()
        self.model = model

        config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model.text_model = peft.LoraModel(self.model.text_model, config, "LoRA")
        self.model.vision_model = peft.LoraModel(self.model.vision_model, config, "LoRA")

        self.model.vision_model = nn.DataParallel(self.model.vision_model, device_ids=gpu_ids, output_device=gpu_ids[0])

    def forward(self, images, input_ids, attention_mask, coocurrence):
        loss, img_embeds, text_embeds = self.model(images, input_ids, attention_mask, coocurrence)
        return loss, img_embeds, text_embeds


class XRay_CLIP(nn.Module):
    def __init__(self, model, gpu_id):
        super(XRay_CLIP, self).__init__()
        self.model = model.to(gpu_id)

        config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model.text_encoder_q = peft.LoraModel(self.model.text_encoder_q, config, "LoRA")
        self.model.img_encoder_q = peft.LoraModel(self.model.img_encoder_q, config, "LoRA")

    def forward(self, imgs, input_ids, attention_mask, token_type_ids,
                center_input_ids, center_attention_mask,
                center_token_type_id):
        loss, img_embeds, text_embeds, center_embeds = self.model(imgs, input_ids, attention_mask, token_type_ids,
                                                                  center_input_ids, center_attention_mask,
                                                                  center_token_type_id)
        return loss, img_embeds, text_embeds, center_embeds


class Pathology_CLIP(nn.Module):
    def __init__(self, model, gpu_id):
        super(Pathology_CLIP, self).__init__()
        self.model = model.to(gpu_id)

        config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model.text_model = peft.LoraModel(self.model.text_model, config, "LoRA")
        self.model.vision_model = peft.LoraModel(self.model.vision_model, config, "LoRA")

    def forward(self, pathology_inputs):
        output = self.model(pathology_inputs, return_loss=True)
        return output.loss, output.image_embeds, output.text_embeds


class ECG_CLIP(nn.Module):
    def __init__(self, model, gpu_id):
        super(ECG_CLIP, self).__init__()
        self.model = model.to(gpu_id)

        config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model.text_model = peft.LoraModel(self.model.text_model, config, "LoRA")
        self.model.vision_model = peft.LoraModel(self.model.vision_model, config, "LoRA")

    def forward(self, imgs, input_ids, attention_mask, token_type_ids):
        loss, img_embeds, text_embeds = self.model(imgs, input_ids, attention_mask, token_type_ids)
        return loss, img_embeds, text_embeds
