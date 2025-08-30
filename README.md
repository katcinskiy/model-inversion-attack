# Model Inversion Attack

Implementation of model inversion attacks using `Qwen2.5-1.5B` as LLM and `bart-base` as inverse model.  

---

## Weights
Pretrained weights for the inversion model: [Google Drive link](https://drive.google.com/drive/folders/10P259HD9siA4foxBeN8c_pIdKeKmWM8R?usp=share_link)

---

## Configs
`./conf` contains 4 configs, each targeting inversion after a different transformer layer.

---

## Example Use
This repo focuses on inversion attacks in general.  
It is also used in my unofficial implementation of [Stained Glass Transformer (SGT)](https://github.com/katcinskiy/stained-glass-transform-pytorch) to test robustness against such attacks.
