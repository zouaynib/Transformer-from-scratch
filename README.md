# transformers-from-scratch

A decoder-only transformer built from first principles in PyTorch — one component at a time,
each with tests and interview notes. Starts from the vanilla GPT block and upgrades every
piece into its modern Mistral-style equivalent.

Written as interview preparation for **Mistral AI**.

## Ground rules

- No `nn.MultiheadAttention`, no `nn.TransformerBlock`, no HuggingFace. Only `nn.Linear`,
  `nn.Embedding`, `nn.Parameter` and raw tensor ops.
- Every part ships with tests that check **correctness**, not just shapes — causality,
  numerical stability, and equivalence against a PyTorch reference where one exists.
- Every part ships with a note in `notes/` covering the questions an interviewer asks about it.

## Roadmap

| # | Part | Module | Status |
|---|------|--------|--------|
| 1 | Scaled dot-product attention | `attention.py` | ✅ |
| 2 | Multi-head attention | `attention.py` | ⬜ |
| 3 | Position-wise feed-forward | `feedforward.py` | ⬜ |
| 4 | Normalization: LayerNorm → RMSNorm | `norm.py` | ⬜ |
| 5 | The transformer block (pre-LN + residuals) | `block.py` | ⬜ |
| 6 | Embeddings, weight tying, learned positions | `embedding.py` | ⬜ |
| 7 | Full GPT model + greedy generation | `model.py` | ⬜ |
| 8 | Tokenizer + training loop on tiny data | `tokenizer.py`, `train.py` | ⬜ |
| 9 | Sampling: temperature, top-k, top-p | `sampling.py` | ⬜ |
| 10 | RoPE — rotary position embeddings | `rope.py` | ⬜ |
| 11 | KV cache — fast autoregressive decoding | `cache.py` | ⬜ |
| 12 | Grouped-Query Attention (GQA) | `attention.py` | ⬜ |
| 13 | SwiGLU feed-forward | `feedforward.py` | ⬜ |
| 14 | Sliding-window attention | `attention.py` | ⬜ |
| 15 | Mixture of Experts (Mixtral-style) | `moe.py` | ⬜ |
| 16 | Assemble a Mistral-shaped model end to end | `mistral.py` | ⬜ |

## Setup

```bash
pip install -r requirements.txt
pytest
```

## Notes

Interview-facing explanations live in [`notes/`](notes/) — one per part.
