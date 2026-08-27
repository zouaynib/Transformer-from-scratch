"""Scaled dot-product attention — the atom every other part is built from.

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Read it as a soft dictionary lookup. Each query is compared against every key by dot
product; the similarities become a probability distribution; the output is the weighted
average of the values under that distribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["scaled_dot_product_attention", "causal_mask"]


def causal_mask(
    seq_len: int,
    key_len: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Lower-triangular boolean mask; ``True`` marks a position a query may attend to.

    With ``key_len > seq_len`` the mask is right-aligned, so query ``i`` sees the first
    ``key_len - seq_len + i + 1`` keys. That is the layout needed once a KV cache holds
    past keys and only the new tokens are passed as queries (Part 11).
    """
    key_len = seq_len if key_len is None else key_len
    offset = key_len - seq_len
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(key_len, device=device).unsqueeze(0)
    return cols <= rows + offset


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute attention and return ``(output, attention_weights)``.

    Args:
        query: ``(..., T_q, d_k)``
        key:   ``(..., T_k, d_k)``
        value: ``(..., T_k, d_v)`` — ``d_v`` need not equal ``d_k``.
        mask: broadcastable boolean tensor over ``(..., T_q, T_k)`` where ``True`` means
            *attend* and ``False`` means *ignore*. A ``(T_k,)`` or ``(1, T_k)`` mask
            applies the same key mask to every query, which is what padding needs.
        is_causal: additionally forbid attending to future positions.
        dropout_p: dropout applied to the attention weights, not to the output.
        training: dropout is a no-op when ``False``.

    Returns:
        output ``(..., T_q, d_v)`` and weights ``(..., T_q, T_k)``.
    """
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query and key must share d_k, got {query.shape[-1]} and {key.shape[-1]}"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key and value must share T_k, got {key.shape[-2]} and {value.shape[-2]}"
        )

    d_k = query.shape[-1]

    # (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    #
    # The 1/√d_k is not cosmetic. With q, k entries of unit variance, each dot product is a
    # sum of d_k independent products, so its variance grows like d_k and its scale like
    # √d_k. Feed that into softmax and for large d_k the distribution saturates onto one
    # key; the softmax Jacobian is then ~0 and no gradient reaches Q or K. Dividing by √d_k
    # holds the logits at unit variance regardless of head size.
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if is_causal:
        cm = causal_mask(query.shape[-2], key.shape[-2], device=query.device)
        mask = cm if mask is None else (mask & cm)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be a bool tensor, got {mask.dtype}")
        # Fill with the dtype's most negative *finite* value rather than -inf. A query row
        # whose keys are all masked (a padded query in a padded batch) would otherwise be
        # -inf everywhere, and softmax of all -inf is NaN, which poisons the backward pass
        # for the whole batch. With a finite floor such a row degrades to a uniform
        # distribution over junk — harmless, because its output is discarded by the loss.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # Subtracting the row max (done inside torch.softmax) is what keeps exp() from
    # overflowing on large logits; softmax is invariant to that shift.
    attn = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        # Dropping attention weights de-correlates the heads and stops any single
        # key-value pair from dominating. Rows no longer sum to 1 during training —
        # that is expected, the 1/(1-p) rescaling keeps the expectation right.
        attn = F.dropout(attn, p=dropout_p, training=training)

    output = attn @ value  # (..., T_q, T_k) @ (..., T_k, d_v) -> (..., T_q, d_v)
    return output, attn
