# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission tests for the guarded Gemma Uno splitKV attention path.

The tests extract the two small admission helpers from the source tree.  They
exercise the real shape, mask, and scratch predicates without importing Triton
or compiling a CUDA kernel; GPU numerical and graph tests remain in the
attention probe.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(
    os.environ.get("VLLM_UNO_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)


def _load_functions(relative_path: str, names: set[str]) -> dict[str, object]:
    tree = ast.parse((SOURCE / relative_path).read_text(encoding="utf-8"))
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "NUM_PAR_SOFTMAX_SEGMENTS"
                for target in node.targets
            )
        )
    ]
    namespace = {
        "os": os,
        "torch": torch,
        "KVQuantMode": SimpleNamespace(NONE=object()),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), relative_path, "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


@pytest.fixture(scope="module")
def static_width():
    return _load_functions(
        "vllm/v1/attention/backends/triton_attn.py",
        {"_uno_static_query_width"},
    )["_uno_static_query_width"]


@pytest.fixture(scope="module")
def admission():
    return _load_functions(
        "vllm/v1/attention/ops/triton_unified_attention.py",
        {"_uno_splitkv_env_enabled", "_uno_gemma_splitkv_admissible"},
    )


def _static_kwargs(**overrides):
    kwargs = dict(
        use_uno=True,
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        causal=True,
        uno_custom_mask=None,
        mm_req_doc_ranges=None,
        rswa_prefix_lens=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_static_width_accepts_only_host_validated_c1_layout(static_width):
    assert static_width(**_static_kwargs()) == 4
    assert static_width(**_static_kwargs(use_uno=False)) is None
    assert static_width(**_static_kwargs(num_reqs=2)) is None
    one_query = dict(
        max_query_len=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
    )
    assert static_width(**_static_kwargs(**one_query)) is None
    six_queries = dict(
        max_query_len=6,
        num_actual_tokens=6,
        query_start_loc_cpu=torch.tensor([0, 6]),
    )
    assert static_width(**_static_kwargs(**six_queries)) is None
    assert (
        static_width(**_static_kwargs(query_start_loc_cpu=torch.tensor([0, 3]))) is None
    )
    assert (
        static_width(**_static_kwargs(query_start_loc_cpu=torch.tensor([0, 4, 4])))
        is None
    )
    assert static_width(**_static_kwargs(causal=torch.tensor([True]))) is None
    assert static_width(**_static_kwargs(mm_req_doc_ranges={})) is None
    assert static_width(**_static_kwargs(rswa_prefix_lens=torch.zeros(1))) is None


def _attention_case(
    head_size=512,
    num_kv_heads=2,
    width=4,
    scratch_rows=8,
    scratch_heads=16,
    scratch_segments=16,
    scratch_head_size=None,
):
    if scratch_head_size is None:
        scratch_head_size = head_size
    tensors = {
        "q": torch.zeros((width, 16, head_size), dtype=torch.bfloat16),
        "k": torch.zeros((1, 16, num_kv_heads, head_size), dtype=torch.bfloat16),
        "v": torch.zeros((1, 16, num_kv_heads, head_size), dtype=torch.bfloat16),
        "out": torch.zeros((width, 16, head_size), dtype=torch.bfloat16),
        "max_seqlen_q": width,
        "num_seqs": 1,
        "causal": True,
        "window_size": (-1, -1) if head_size == 512 else (1023, 0),
        "uno_static_query_width": width,
        "seq_threshold_3D": scratch_rows,
        "num_par_softmax_segments": scratch_segments,
        "softmax_segm_output": torch.empty(
            (scratch_rows, scratch_heads, scratch_segments, scratch_head_size),
            dtype=torch.float32,
        ),
        "softmax_segm_max": torch.empty(
            (scratch_rows, scratch_heads, scratch_segments), dtype=torch.float32
        ),
        "softmax_segm_expsum": torch.empty(
            (scratch_rows, scratch_heads, scratch_segments), dtype=torch.float32
        ),
        "mm_prefix_range": None,
        "rswa_prefix_lens": None,
        "rswa_window": None,
        "alibi_slopes": None,
        "qq_bias": None,
        "sinks": None,
        "output_scale": None,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "kv_quant_mode": SimpleNamespace(NONE=object()).NONE,
        "chunk_lookback": -1,
    }
    return tensors


def _none_kv_mode(admission):
    """The module's own KVQuantMode.NONE, which the helper compares against."""
    globals_ = admission["_uno_gemma_splitkv_admissible"].__globals__
    return globals_["KVQuantMode"].NONE


def _admitted(admission, **overrides):
    shape_keys = {
        "head_size",
        "num_kv_heads",
        "width",
        "scratch_rows",
        "scratch_heads",
        "scratch_segments",
        "scratch_head_size",
    }
    args = _attention_case(
        **{key: value for key, value in overrides.items() if key in shape_keys}
    )
    args.update(
        {key: value for key, value in overrides.items() if key not in shape_keys}
    )
    # The source helper compares against its module's enum singleton.  Replace
    # the fixture value with that exact singleton for the normal case.
    args["kv_quant_mode"] = _none_kv_mode(admission)
    return admission["_uno_gemma_splitkv_admissible"](**args)


@pytest.mark.parametrize(
    "head_size,num_kv_heads",
    [(256, 8), (512, 2)],
)
@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_admission_accepts_both_gemma_bf16_families(
    admission, head_size, num_kv_heads, width
):
    assert _admitted(
        admission, head_size=head_size, num_kv_heads=num_kv_heads, width=width
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_size": 128, "num_kv_heads": 8},
        {"head_size": 256, "num_kv_heads": 2},
        {"head_size": 512, "num_kv_heads": 8},
        {"scratch_rows": 3},
        {"scratch_heads": 8},
        {"scratch_segments": 8},
        {"scratch_head_size": 128},
        {"mm_prefix_range": torch.zeros(1)},
        {"rswa_prefix_lens": torch.zeros(1)},
        {"rswa_window": 128},
        {"causal": torch.tensor([True])},
        {"chunk_lookback": 1},
    ],
)
def test_admission_fails_closed_for_shape_mask_and_scratch_mismatches(
    admission, overrides
):
    assert not _admitted(admission, **overrides)


def test_admission_rejects_noncontiguous_raw_scratch(admission):
    args = _attention_case(head_size=256, num_kv_heads=8, scratch_head_size=512)
    base = torch.empty((8, 16, 16, 512), dtype=torch.float32)
    args["softmax_segm_output"] = base[..., ::2]
    args["kv_quant_mode"] = _none_kv_mode(admission)
    assert not admission["_uno_gemma_splitkv_admissible"](**args)


def test_none_kv_mode_allows_inert_backend_scales(admission):
    # TritonAttentionImpl materializes layer K/V scales even for BF16
    # KVQuantMode.NONE.  They are ignored by the kernel in this mode.
    assert _admitted(
        admission,
        k_descale=torch.ones(1),
        v_descale=torch.ones(1),
    )


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_environment_is_explicit_opt_in(admission, monkeypatch, value):
    monkeypatch.setenv("UNO_GEMMA_SPLITKV", value)
    assert admission["_uno_splitkv_env_enabled"]()


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "random"])
def test_environment_defaults_off(admission, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("UNO_GEMMA_SPLITKV", raising=False)
    else:
        monkeypatch.setenv("UNO_GEMMA_SPLITKV", value)
    assert not admission["_uno_splitkv_env_enabled"]()


def test_c1_marker_refuses_multi_request_layout(static_width):
    # The legacy resolver addresses B=2,width=4,BLOCK_Q=2 as [0,1,3,4]; a
    # contiguous exact-grid optimization would silently drop block 4.
    assert (
        static_width(
            **_static_kwargs(
                num_reqs=2,
                num_actual_tokens=8,
                query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
            )
        )
        is None
    )
