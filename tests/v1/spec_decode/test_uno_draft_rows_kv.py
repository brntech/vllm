# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Uno draft-rows contract on the shared attention layers.

The draft rows reuse the target's attention layers and KV cache, so the
arrangement must be a causal full KV allocation and every draft layer must run
on a backend the draft can drive. Sliding-window checkpoints (e.g. Gemma with
mixed local/global layers) are admitted only under that full allocation, where
each layer's own kernel bounds its window and no window-owned row can be
recycled under a draft write.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.spec_decode.uno import UnoSpeculator

KB = 1024


def _full_attention(window: int | None = None, **overrides) -> FullAttentionSpec:
    kwargs = dict(
        block_size=16,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        sliding_window=window,
    )
    kwargs.update(overrides)
    return FullAttentionSpec(**kwargs)


def _uniform(specs: dict[str, FullAttentionSpec]) -> UniformTypeKVCacheSpecs:
    return UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs)


@pytest.fixture
def proposer():
    instance = object.__new__(UnoSpeculator)
    instance.draft_attn_layer_names = {"layer.0.self_attn.attn"}
    return instance


def _set_attn(proposer, groups):
    kv_cache_config = SimpleNamespace(kv_cache_groups=groups)

    def base_set_attn(self, *_args, **_kwargs):
        self.attn_groups = [
            [SimpleNamespace(supports_draft_decode_metadata_update=True)]
        ]

    with patch(
        "vllm.v1.worker.gpu.spec_decode.speculator.DraftModelSpeculator.set_attn",
        base_set_attn,
    ):
        UnoSpeculator.set_attn(
            proposer,
            SimpleNamespace(),
            kv_cache_config,
            SimpleNamespace(),
            SimpleNamespace(),
            [[SimpleNamespace()]],
        )


def _group(spec):
    return SimpleNamespace(layer_names=["layer.0.self_attn.attn"], kv_cache_spec=spec)


def test_full_attention_group_is_admitted(proposer):
    _set_attn(proposer, [_group(_full_attention())])


def test_promoted_sliding_window_group_is_admitted(proposer):
    """Full allocation promotes local layers to full-attention KV specs."""
    spec = _uniform(
        {
            "layer.0.self_attn.attn": _full_attention(),
            "layer.1.self_attn.attn": _full_attention(),
        }
    )
    _set_attn(proposer, [_group(spec)])


@pytest.mark.parametrize(
    "spec",
    [
        _full_attention(non_causal=True),
        _full_attention(attention_chunk_size=KB),
        SlidingWindowSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            sliding_window=KB,
        ),
        _uniform({"layer.0.self_attn.attn": _full_attention(non_causal=True)}),
    ],
)
def test_rolling_or_non_causal_allocation_is_refused(proposer, spec):
    with pytest.raises(ValueError, match="causal full KV allocation"):
        _set_attn(proposer, [_group(spec)])


def test_multiple_kv_groups_are_refused(proposer):
    with pytest.raises(ValueError, match="one KV cache group"):
        _set_attn(
            proposer,
            [_group(_full_attention()), _group(_full_attention())],
        )


def _layer(backend: str):
    layer = SimpleNamespace(
        get_kv_cache_spec=lambda _config: object(),
        get_attn_backend=lambda: SimpleNamespace(get_name=lambda: backend),
    )
    return layer


def _load_model(proposer, backend: str):
    layers = {"layer.0.self_attn.attn": _layer(backend)}
    proposer.vllm_config = SimpleNamespace(speculative_config=SimpleNamespace())
    proposer.use_local_argmax_reduction = False
    with patch(
        "vllm.v1.worker.gpu.spec_decode.uno.get_layers_from_vllm_config",
        return_value=layers,
    ):
        UnoSpeculator.load_model(proposer, SimpleNamespace())


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "TRITON_ATTN"])
def test_admitted_draft_backends(proposer, backend):
    _load_model(proposer, backend)
    assert proposer.draft_attn_layer_names == {"layer.0.self_attn.attn"}


def test_unknown_draft_backend_is_refused(proposer):
    with pytest.raises(ValueError, match="FLASH_ATTN or TRITON_ATTN"):
        _load_model(proposer, "FLEX_ATTENTION")
