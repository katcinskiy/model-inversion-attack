import gc

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

class BlackBoxModel(nn.Module):
    def __init__(self, model, tokenizer, layer_embeds_to_return):
        super().__init__()
        self.layer_embeds_to_return = layer_embeds_to_return

        self.tokenizer = tokenizer
        self.model = model

        self.model.eval()

        # remove all layers after layer_embeds_to_return, because we don't need them
        K = self.layer_embeds_to_return
        layers = self.model.model.layers
        del layers[K:]
        self.model.model.config.num_hidden_layers = K

        gc.collect()
        torch.cuda.empty_cache()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()   # never let the inner model leave eval
        return self
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        result = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        return result.hidden_states[self.layer_embeds_to_return]

class InversionModel(nn.Module):
    def __init__(self, model, tokenizer, input_features_d):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model

        self.proj = nn.Linear(input_features_d, model.model.encoder.embed_tokens.embedding_dim)

        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def forward(self, encoder_embeds, encoder_attention_mask, labels, **kwargs):
        transformed_embeds = self.proj(encoder_embeds)

        return self.model(
            inputs_embeds=transformed_embeds, 
            attention_mask=encoder_attention_mask, 
            labels=labels,
            **kwargs
        )


class WrapperModel(nn.Module):
    def __init__(self, bb_model, inversion_model):
        super().__init__()
        self.bb_model = bb_model
        self.inversion_model = inversion_model

    def forward(self, bb_input_ids, bb_attention_mask, inv_input_ids, inv_attention_mask, labels=None):
        # with torch.autocast(device_type="cuda"):
        embeds_after_X_layers = self.bb_model(input_ids=bb_input_ids, attention_mask=bb_attention_mask)

        return self.inversion_model(
            encoder_embeds=embeds_after_X_layers,
            encoder_attention_mask=bb_attention_mask,
            labels=labels,
        )
