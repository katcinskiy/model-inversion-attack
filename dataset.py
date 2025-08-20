import torch

from torch.nn.utils.rnn import pad_sequence

class InvAttackDataset(torch.utils.data.Dataset):
    def __init__(self, texts, bb_tokenizer, inv_tokenizer, max_length):
        super().__init__()
        self.bb_tokenizer = bb_tokenizer
        self.inv_tokenizer = inv_tokenizer
        self.max_length = max_length

        self.texts = texts

    def __getitem__(self, idx):
        attacked_model_tokenized = self.bb_tokenizer(self.texts[idx], return_tensors='pt', truncation=True, max_length=self.max_length)
        base_model_tokenized = self.inv_tokenizer(self.texts[idx], return_tensors='pt', truncation=True, max_length=self.max_length)

        return {
            "text": self.texts[idx],
            "bb_input_ids": attacked_model_tokenized['input_ids'][0],
            "bb_attention_mask": attacked_model_tokenized['attention_mask'][0],
            "inv_input_ids": base_model_tokenized['input_ids'][0],
            "inv_attention_mask": base_model_tokenized['attention_mask'][0],
        }

    def __len__(self):
        return len(self.texts)
    

def make_collate_fn(bb_tokenizer, inv_tokenizer):
    bb_pad_token = bb_tokenizer.pad_token_id
    inv_pad_token = inv_tokenizer.pad_token_id

    def collate_fn(batch):
        bb_input_ids = [x["bb_input_ids"] for x in batch]
        inv_input_ids = [x["inv_input_ids"] for x in batch]

        bb_input_ids = pad_sequence(bb_input_ids, batch_first=True, padding_value=bb_pad_token)
        inv_input_ids = pad_sequence(inv_input_ids, batch_first=True, padding_value=inv_pad_token)

        bb_attention_mask = [x["bb_attention_mask"] for x in batch]
        inv_attention_mask = [x["inv_attention_mask"] for x in batch]

        bb_attention_mask = pad_sequence(bb_attention_mask, batch_first=True, padding_value=0)
        inv_attention_mask = pad_sequence(inv_attention_mask, batch_first=True, padding_value=0)

        # labels for HF CE loss: ignore pads
        labels = inv_input_ids.masked_fill(inv_attention_mask == 0, -100)

        return {
            "bb_input_ids": bb_input_ids,
            "bb_attention_mask": bb_attention_mask,

            "labels": labels.long(),

            "inv_input_ids": inv_input_ids,
            "inv_attention_mask": inv_attention_mask,
        }
    
    return collate_fn
