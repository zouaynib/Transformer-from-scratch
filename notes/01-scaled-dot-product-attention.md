# Part 1 — Scaled dot-product attention

$$\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Shapes: `Q (T_q, d_k)`, `K (T_k, d_k)`, `V (T_k, d_v)` → output `(T_q, d_v)`.
Leading batch/head dimensions broadcast.

## The one-sentence intuition

A differentiable dictionary lookup. A hard lookup would be `argmax` over key matches;
softmax replaces the argmax with a temperature-controlled soft selection so the whole
thing has gradients.

## Why divide by √d_k

Take `q, k ∈ ℝ^{d_k}` with i.i.d. zero-mean unit-variance entries. Then

$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i,\qquad \mathbb{E}[q\cdot k]=0,\qquad \mathrm{Var}(q\cdot k)=d_k$$

so the logits have standard deviation `√d_k`. For `d_k = 128` the logits span roughly
±30, softmax saturates into a one-hot, and its Jacobian

$$\frac{\partial\, \mathrm{softmax}(x)_i}{\partial x_j} = p_i(\delta_{ij} - p_j)$$

goes to zero because `p_i(1-p_i) → 0`. Gradients to Q and K vanish. Dividing by `√d_k`
restores unit-variance logits for any head size — that is the entire reason the term
exists, and it is why the scale is `√d_k` and not `d_k` or `√d_model`.

**Follow-up they like:** *why not just LayerNorm the logits?* Because the scale is known
in closed form, costs nothing, and adds no parameters. LayerNorm would work but is a
learned solution to a solved problem.

## Why mask before the softmax, not zero the weights after

Masking after softmax leaves the masked keys contributing to the denominator, so the
surviving weights no longer sum to 1 — a silent, sequence-length-dependent rescaling of
every output. Setting the logits to `-∞` removes those terms from the normaliser itself.

## The `-inf` trap

`masked_fill(~mask, float("-inf"))` produces `NaN` the moment a query row is fully
masked — a padded query position in a padded batch. `softmax([-inf, -inf])` is `0/0`.
The `NaN` then propagates through the backward pass and destroys the whole batch, not
just that row. Fix: fill with `torch.finfo(dtype).min` instead. The row degrades to a
uniform distribution over garbage, which is fine because the loss masks it out anyway.

This is the single most common bug in a from-scratch implementation. It only shows up
with padding, so it survives every toy test.

## Numerical stability of softmax

`torch.softmax` subtracts the row max before exponentiating:

$$\mathrm{softmax}(x)_i = \frac{e^{x_i - \max x}}{\sum_j e^{x_j - \max x}}$$

Softmax is shift-invariant, so this is exact, not an approximation — it just keeps
`exp()` from overflowing. Under `float16`, whose max is 65504, `exp(12)` already
overflows, which is why attention logits are commonly kept in fp32 under AMP.

## Complexity

| | Time | Memory |
|---|---|---|
| `QKᵀ` | `O(T² d_k)` | `O(T²)` for the score matrix |
| `softmax` | `O(T²)` | — |
| `× V` | `O(T² d_v)` | — |

Quadratic in sequence length in both — the `T²` score matrix is what makes long context
expensive and what FlashAttention removes by never materialising it (it tiles the
computation and keeps a running softmax normaliser in SRAM). Same math, same output,
`O(T)` memory instead of `O(T²)`. Sliding-window attention (Part 14) attacks the same
cost from the other side, by making the mask band-limited.

## Causality

The mask is lower-triangular: query `i` sees keys `0..i`. That single constraint is what
makes teacher forcing valid — every position can be trained in parallel on the true
prefix, because no position can see its own future. `test_changing_a_future_token_cannot_
change_an_earlier_output` is the test that actually proves it; a shape test does not.

The mask is right-aligned when `T_k > T_q`, which is the layout needed once a KV cache
holds the past and only new tokens arrive as queries.

## Where dropout goes

On the attention **weights**, after softmax. It de-correlates heads and stops one
key-value pair from dominating. Rows stop summing to 1 during training — expected, since
the `1/(1-p)` rescaling preserves the expectation.

## Traps checklist

- [ ] Scale by `√d_k`, not `d_k`, and not `√d_model`.
- [ ] Transpose the *last two* dims of K, not `.T` (which fails on batched tensors).
- [ ] Mask before softmax.
- [ ] Use `finfo.min`, not `-inf`.
- [ ] `d_v` need not equal `d_k`.
- [ ] `softmax(dim=-1)` — over keys. `dim=-2` trains and silently learns nothing useful.
