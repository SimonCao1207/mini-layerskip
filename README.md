# mini-layerskip


```plaintext
[Input Tokens] 
       │
       ▼
[Transformer Layers 1 ... k]  <-- Draft Stage (exit_layer = k)
       │
 ┌────[Draft Tokens]────┐
 │                      │
 │   (Draft Decoding)   │
 ▼                      │
[Transformer Layers (k+1) ... N]
       │
       │
       └─── Verification Stage (Full forward pass) ────► Final Output
```

The purpose of this repo is to replicate the approach from the paper [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710), which proposes an end-to-end pipeline for self-speculative decoding with early exit. Here, we aim to test a similar strategy applied to pretrained models, focusing solely on its effectiveness during inference without any training.