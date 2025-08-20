import torch

from transformers import TrainerCallback

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, bb_tokenizer, inv_tokenizer, samples, max_new_tokens=32):
        super().__init__()
        self.bb_tokenizer = bb_tokenizer
        self.inv_tokenizer = inv_tokenizer
        self.samples = samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        device = args.device if hasattr(args, "device") else "cuda"

        print("\n--- Generation check ---")
        for text in self.samples:
            bb_inputs = self.bb_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_new_tokens
            ).to(device)

            with torch.no_grad():
                embeds = model.bb_model(**bb_inputs)

            with torch.no_grad():
                generated_ids = model.inversion_model.model.generate(
                    inputs_embeds=model.inversion_model.proj(embeds),
                    attention_mask=bb_inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=3
                )

            decoded = self.inv_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"Original:     {text}")
            print(f"Reconstructed: {decoded[0]}")
            print("-" * 50)

        return control
