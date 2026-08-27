import math

import pytest
import torch
import torch.nn.functional as F

from scratchformer.attention import causal_mask, scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _qkv(B=2, H=3, T=5, d=8, Tk=None):
    Tk = T if Tk is None else Tk
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, Tk, d)
    v = torch.randn(B, H, Tk, d)
    return q, k, v


# --- shapes -----------------------------------------------------------------

def test_output_shape_matches_query_length_and_value_dim():
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 7, 8)
    v = torch.randn(2, 3, 7, 16)  # d_v may differ from d_k
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 5, 16)
    assert attn.shape == (2, 3, 5, 7)


def test_works_without_batch_or_head_dims():
    q, k, v = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (4, 8)
    assert attn.shape == (4, 4)


# --- the softmax is a real distribution -------------------------------------

def test_attention_rows_sum_to_one():
    q, k, v = _qkv()
    _, attn = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-6)
    assert (attn >= 0).all()


# --- the scaling factor is actually applied ---------------------------------

def test_scaling_by_sqrt_dk():
    q, k, v = _qkv(B=1, H=1, T=4, d=16)
    _, attn = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(16), dim=-1)
    assert torch.allclose(attn, expected, atol=1e-6)


def test_unscaled_would_differ_so_the_test_above_is_not_vacuous():
    q, k, v = _qkv(B=1, H=1, T=4, d=16)
    _, attn = scaled_dot_product_attention(q, k, v)
    unscaled = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
    assert not torch.allclose(attn, unscaled, atol=1e-4)


# --- causality --------------------------------------------------------------

def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    assert m.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_causal_attention_puts_zero_weight_on_the_future():
    q, k, v = _qkv(T=6)
    _, attn = scaled_dot_product_attention(q, k, v, is_causal=True)
    future = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert attn.masked_select(future).abs().max() == 0.0


def test_changing_a_future_token_cannot_change_an_earlier_output():
    """The property that makes teacher forcing valid. If this breaks, the LM cheats."""
    q, k, v = _qkv(T=6)
    out_a, _ = scaled_dot_product_attention(q, k, v, is_causal=True)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, 4:] = torch.randn_like(k2[:, :, 4:])
    v2[:, :, 4:] = torch.randn_like(v2[:, :, 4:])
    out_b, _ = scaled_dot_product_attention(q, k2, v2, is_causal=True)

    assert torch.allclose(out_a[:, :, :4], out_b[:, :, :4], atol=1e-6)
    assert not torch.allclose(out_a[:, :, 4:], out_b[:, :, 4:], atol=1e-6)


# --- explicit masks ---------------------------------------------------------

def test_boolean_mask_false_means_do_not_attend():
    q, k, v = _qkv(B=1, H=1, T=3)
    mask = torch.tensor([[True, False, True]])  # broadcasts over query positions
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert attn[..., 1].abs().max() == 0.0
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-6)


def test_mask_and_is_causal_combine():
    q, k, v = _qkv(B=1, H=1, T=4)
    mask = torch.tensor([[True, True, False, True]])
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=True)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert attn.masked_select(future).abs().max() == 0.0
    assert attn[..., 2].abs().max() == 0.0


def test_fully_masked_row_does_not_produce_nan():
    """A padded query row can have every key masked. -inf everywhere -> NaN, unless guarded."""
    q, k, v = _qkv(B=1, H=1, T=2)
    mask = torch.tensor([[False, False], [True, True]])
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()


# --- equivalence with the PyTorch reference ---------------------------------

@pytest.mark.parametrize("is_causal", [False, True])
def test_matches_torch_reference(is_causal):
    q, k, v = _qkv()
    ours, _ = scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_matches_torch_reference_with_a_mask():
    q, k, v = _qkv()
    mask = torch.rand(5, 5) > 0.3
    mask.fill_diagonal_(True)  # keep every row non-empty
    ours, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(ours, ref, atol=1e-6)


# --- numerical behaviour ----------------------------------------------------

def test_large_logits_stay_finite():
    q, k, v = _qkv(B=1, H=1, T=4, d=8)
    out, attn = scaled_dot_product_attention(q * 1e4, k * 1e4, v)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()


def test_uniform_keys_give_uniform_attention():
    q = torch.randn(1, 1, 3, 8)
    k = torch.zeros(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(attn, torch.full_like(attn, 0.25), atol=1e-6)
    assert torch.allclose(out, v.mean(dim=-2, keepdim=True).expand_as(out), atol=1e-6)


# --- dropout & gradients ----------------------------------------------------

def test_dropout_only_applies_in_training():
    q, k, v = _qkv()
    a, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=False)
    b, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_gradients_flow_to_all_three_inputs():
    q, k, v = _qkv()
    for t in (q, k, v):
        t.requires_grad_(True)
    out, _ = scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()
