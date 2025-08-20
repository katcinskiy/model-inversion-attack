import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



import torch
from torch.utils.data import random_split

from models import WrapperModel, InversionModel, BlackBoxModel

from callbacks import GenerationEvalCallback

from dataset import InvAttackDataset, make_collate_fn

from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset

device = torch.device('cuda:0')

USE_DATASETS = True
DATASET_NAME = "ag_news"
N_SAMPLES = 10000
LR = 5e-5
EPOCHS = 100
BATCH_SIZE = 32

LAYER_EMBEDS_TO_RETURN = 1

MAX_LENGTH = 64

inverse_model_name = "facebook/bart-base"
bb_model_name = "Qwen/Qwen2.5-1.5B"

bb_tokenizer = AutoTokenizer.from_pretrained(bb_model_name)
bb_model = Qwen2ForCausalLM.from_pretrained(bb_model_name).to(device)
bb_model = BlackBoxModel(bb_model, bb_tokenizer, layer_embeds_to_return=LAYER_EMBEDS_TO_RETURN)

inv_tokenizer = BartTokenizer.from_pretrained(inverse_model_name)
inv_model = BartForConditionalGeneration.from_pretrained(inverse_model_name).eval().to(device)
inv_model = InversionModel(inv_model, inv_tokenizer, input_features_d=1536)

ds = load_dataset(DATASET_NAME, split="train").select(range(N_SAMPLES))
texts = [item['text'] for item in ds]

dataset = InvAttackDataset(texts, bb_tokenizer, inv_tokenizer, max_length=MAX_LENGTH)
lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]

train_dataset, eval_dataset = random_split(dataset, lengths)

eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(len(eval_dataset), 500)))

model = WrapperModel(bb_model, inv_model)


training_args = TrainingArguments(
    output_dir="./bb_inversion_attack_trainer",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    save_safetensors=False,
    save_total_limit=1
)

# proj_params = [p for n, p in model.named_parameters() if n.startswith("proj.")]
proj_params = list(model.inversion_model.proj.parameters())
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
optimizer = torch.optim.AdamW([
    {"params": proj_params, "lr": 1e-3},
    {"params": lora_params, "lr": 3e-4},
], weight_decay=0.0)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=make_collate_fn(bb_tokenizer, inv_tokenizer),
    optimizers=(optimizer, None)
)

texts = [
    eval_dataset[0]['text'],
    eval_dataset[1]['text'],
    eval_dataset[2]['text'],
]

trainer.add_callback(GenerationEvalCallback(bb_tokenizer, inv_tokenizer, texts, max_new_tokens=32))

trainer.train(resume_from_checkpoint=False)
