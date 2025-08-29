import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    device = torch.device('cuda:0')

    bb_model_name = cfg.models.bb_model_name
    inverse_model_name = cfg.models.inverse_model_name
    layer_embeds_to_return = cfg.models.layer_embeds_to_return
    dataset_size = cfg.training.dataset_size
    dataset_name = cfg.training.dataset_name
    max_length = cfg.training.max_length
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    lr = cfg.training.lr

    bb_tokenizer = AutoTokenizer.from_pretrained(bb_model_name)
    bb_model = Qwen2ForCausalLM.from_pretrained(bb_model_name).to(device)
    bb_model = BlackBoxModel(bb_model, bb_tokenizer, layer_embeds_to_return=layer_embeds_to_return)

    inv_tokenizer = BartTokenizer.from_pretrained(inverse_model_name)
    inv_model = BartForConditionalGeneration.from_pretrained(inverse_model_name).eval().to(device)
    inv_model = InversionModel(inv_model, inv_tokenizer, input_features_d=1536)

    ds = load_dataset(dataset_name, split="train").select(range(dataset_size))
    texts = [item['text'] for item in ds]

    dataset = InvAttackDataset(texts, bb_tokenizer, inv_tokenizer, max_length=max_length)
    lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]

    train_dataset, eval_dataset = random_split(dataset, lengths)

    eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(len(eval_dataset), 500)))

    model = WrapperModel(bb_model, inv_model)

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        save_safetensors=False,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        warmup_steps=500
    )

    # proj_params = [p for n, p in model.named_parameters() if n.startswith("proj.")]
    proj_params = list(model.inversion_model.proj.parameters())
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": lr * 10},
        {"params": lora_params, "lr": lr},
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


if __name__ == '__main__':
    main()