# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the opt-in draft MoE top-k candidate."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(
    os.environ.get("VLLM_UNO_SOURCE_ROOT", Path(__file__).resolve().parents[3])
)
MODULE_PATH = SOURCE / "vllm/v1/worker/gpu/spec_decode/uno_draft_moe.py"
_spec = importlib.util.spec_from_file_location("uno_draft_moe_under_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
moe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = moe
_spec.loader.exec_module(moe)

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (  # noqa: E402
    MarlinExperts as _MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (  # noqa: E402
    MoEPrepareAndFinalizeNoDPEPModular as _NoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (  # noqa: E402
    CustomRoutingRouter as _CustomRoutingRouter,
)


class CustomRoutingRouter(_CustomRoutingRouter):
    """The real router family; a same-named unrelated class must not pass."""

    def __init__(self):
        super().__init__(
            top_k=8,
            global_num_experts=128,
            custom_routing_function=lambda **kwargs: None,
        )
        self._routing_replay_out = None


class MarlinExperts(_MarlinExperts):
    routing_replay_capture_fn = None
    _routing_replay_buffer = None

    def __init__(self):
        pass

    def workspace_shapes(self, *args):
        return args


class MoEPrepareAndFinalizeNoDPEPModular(_NoDPEPModular):
    def max_num_tokens_per_rank(self):
        return None


class FusedKernel:
    def __init__(self):
        self.fused_experts = MarlinExperts()
        self.prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()


class QuantMethod:
    is_monolithic = False

    def __init__(self):
        self.moe_kernel = FusedKernel()


class MoEConfig:
    experts_per_token = 8
    num_experts = 128

    def __init__(self):
        self.moe_parallel_config = NS(
            tp_size=1,
            pcp_size=1,
            ep_size=1,
            dp_size=1,
            sp_size=1,
            use_ep=False,
            use_all2all_kernels=False,
            enable_eplb=False,
        )


class RoutedExperts:
    top_k = 8

    def __init__(self, config):
        self.moe_config = config
        self.quant_method = QuantMethod()


class MoERunner:
    def __init__(self):
        self.router = CustomRoutingRouter()
        self.moe_config = MoEConfig()
        self.routed_experts = RoutedExperts(self.moe_config)


class RegisteredGemma4Model:
    """The class the model registry resolves the loaded architecture to."""

    config = NS(model_type="gemma4_text", architectures=["Gemma4ForCausalLM"])

    def __init__(self):
        self.moe = MoERunner()

    def named_modules(self):
        yield "", self
        yield "model.layers.0.mlp.experts", self.moe


class Gemma4ForCausalLM:
    """An unrelated class named identically to the Gemma 4 implementation.

    It carries the exact configuration markers and the routing attributes the
    candidate expects, but the registry does not resolve the architecture to
    it, so identity must refuse it.
    """

    config = NS(model_type="gemma4_text", architectures=["Gemma4ForCausalLM"])

    def __init__(self):
        self.moe = MoERunner()

    def named_modules(self):
        yield "", self
        yield "model.layers.0.mlp.experts", self.moe


class LlamaForCausalLM:
    """A non-Gemma implementation whose configuration text mentions gemma4."""

    def __init__(self):
        self.config = NS(model_type="llama", architectures=["LlamaForCausalLM"])
        self.model_config = NS(
            hf_config=NS(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                unrelated_note="converted beside a gemma4 checkpoint",
            )
        )
        self.moe = MoERunner()

    def named_modules(self):
        yield "", self
        yield "model.layers.0.mlp.experts", self.moe


@pytest.fixture(autouse=True)
def reset_process_variant():
    moe._reset_process_variant_for_tests()
    yield
    moe._reset_process_variant_for_tests()


@pytest.fixture
def runtime_moe():
    """The module the proposer imports, with its process state reset."""
    from vllm.v1.worker.gpu.spec_decode import uno_draft_moe as module

    module._reset_process_variant_for_tests()
    yield module
    module._reset_process_variant_for_tests()


@pytest.fixture
def registered_cls():
    """The resolution the loader's model registry would return."""
    return RegisteredGemma4Model


@pytest.fixture
def proposer(monkeypatch, registered_cls):
    monkeypatch.setattr(moe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(moe.torch.cuda, "get_device_capability", lambda device: (8, 6))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda model_config: (registered_cls, "Gemma4ForCausalLM"),
    )
    from vllm.config.compilation import CUDAGraphMode

    model = registered_cls()
    vllm_config = NS(
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=NS(hf_config=NS(architectures=["Gemma4ForCausalLM"])),
    )
    return NS(
        model=model,
        device=torch.device("cuda"),
        vllm_config=vllm_config,
        speculative_config=NS(use_uno=lambda: True),
        cudagraph_manager=NS(cudagraph_mode=CUDAGraphMode.FULL),
        lora_request=NS(lora_int_id=1),
    )


def test_disabled_scope_does_not_import_or_mutate(proposer, monkeypatch):
    monkeypatch.delenv(moe.ENV_NAME, raising=False)
    router = proposer.model.moe.router
    with moe.draft_moe_capture_scope(proposer) as state:
        assert state is None
        assert router.top_k == 8
    assert router.top_k == 8


def test_active_scope_validates_sm86_marlin_and_restores(proposer, monkeypatch):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    router = proposer.model.moe.router
    routed = proposer.model.moe.routed_experts
    with moe.draft_moe_capture_scope(proposer) as state:
        assert state is not None
        assert state.variant == "gemma4-sm86-marlin-topk4"
        assert router.top_k == 4
        assert routed.top_k == 8
        assert state.receipt()[0]["top_k"] == 4
        assert state.receipt()[0]["backend"] == "MarlinExperts"
    assert router.top_k == 8


def test_active_scope_restores_after_capture_error(proposer, monkeypatch):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    router = proposer.model.moe.router
    with (
        pytest.raises(RuntimeError, match="capture failure"),
        moe.draft_moe_capture_scope(proposer),
    ):
        assert router.top_k == 4
        raise RuntimeError("capture failure")
    assert router.top_k == 8


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            lambda p: setattr(p, "cudagraph_manager", NS(cudagraph_mode=None)),
            "captured Uno draft graphs",
        ),
        (lambda p: setattr(p, "lora_request", None), "adapter routing"),
        (
            lambda p: setattr(
                p.model.moe.routed_experts.quant_method.moe_kernel,
                "fused_experts",
                NS(),
            ),
            "backend",
        ),
        (
            lambda p: setattr(
                p.model.moe,
                "moe_config",
                NS(experts_per_token=4, num_experts=128),
            ),
            "configured at 8",
        ),
    ],
)
def test_contract_failures_are_fail_closed(proposer, monkeypatch, change, reason):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    change(proposer)
    with pytest.raises(moe.DraftMoEConfigurationError, match=reason):
        moe.validate_draft_moe(proposer)
    assert proposer.model.moe.router.top_k == 8


def test_variant_cannot_be_removed_after_opt_in(proposer, monkeypatch):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    with moe.draft_moe_capture_scope(proposer):
        pass
    monkeypatch.delenv(moe.ENV_NAME)
    with pytest.raises(moe.DraftMoEConfigurationError, match="enabled|process-fixed"):
        moe.uno_draft_moe_enabled()


def test_unknown_moe_parallelism_fails_closed(proposer, monkeypatch):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    del proposer.model.moe.moe_config.moe_parallel_config.pcp_size
    with pytest.raises(
        moe.DraftMoEConfigurationError, match="cannot identify pcp_size"
    ):
        moe.validate_draft_moe(proposer)


def test_capture_enters_the_scope_only_around_the_draft_graphs():
    """The variant must be confined to the draft capture, never serving."""
    source = (SOURCE / "vllm/v1/worker/gpu/spec_decode/uno.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnoSpeculator"
    )
    capture = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture"
    )
    called = {
        node.func.id
        for node in ast.walk(capture)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "draft_moe_capture_scope" in called
    assert "log_capture_receipt" in source


def test_identity_accepts_the_registered_implementation(proposer, monkeypatch):
    """The class the registry resolves the architecture to passes."""
    monkeypatch.setenv(moe.ENV_NAME, "4")
    moe.validate_draft_moe(proposer)


def test_identity_rejects_an_unrelated_same_named_gemma4_class(proposer, monkeypatch):
    """A same-named class with the exact markers must not pass by name."""
    monkeypatch.setenv(moe.ENV_NAME, "4")
    proposer.model = Gemma4ForCausalLM()
    with pytest.raises(
        moe.DraftMoEConfigurationError, match="registered Gemma 4 implementation"
    ):
        moe.validate_draft_moe(proposer)


def test_identity_requires_the_loaded_gemma4_implementation(proposer, monkeypatch):
    """A non-Gemma implementation must not be admitted by configuration text."""
    monkeypatch.setenv(moe.ENV_NAME, "4")
    proposer.model = LlamaForCausalLM()
    with pytest.raises(
        moe.DraftMoEConfigurationError, match="registered Gemma 4 implementation"
    ):
        moe.validate_draft_moe(proposer)


def test_identity_rejects_non_gemma4_markers_on_a_gemma_implementation(
    proposer, monkeypatch
):
    """Only exact Gemma 4 identifiers qualify; unrelated fields never do."""
    monkeypatch.setenv(moe.ENV_NAME, "4")
    proposer.model.config = NS(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        unrelated_note="converted beside a gemma4 checkpoint",
    )
    proposer.vllm_config.model_config = NS(
        hf_config=NS(architectures=["LlamaForCausalLM"])
    )
    with pytest.raises(moe.DraftMoEConfigurationError, match="exact Gemma 4"):
        moe.validate_draft_moe(proposer)


def test_identity_refuses_when_the_registry_cannot_resolve_the_architecture(
    proposer, monkeypatch
):
    """An unresolvable architecture must never fall through to a name."""
    monkeypatch.setenv(moe.ENV_NAME, "4")

    def unresolvable(model_config):
        raise ValueError("no registered implementation")

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        unresolvable,
    )
    with pytest.raises(moe.DraftMoEConfigurationError, match="cannot resolve"):
        moe.validate_draft_moe(proposer)


def test_identity_rejects_a_same_named_unrelated_router(proposer, monkeypatch):
    """The router contract binds to the class, not its name."""
    monkeypatch.setenv(moe.ENV_NAME, "4")

    class CustomRoutingRouter:
        def __init__(self):
            self.top_k = 8
            self.custom_routing_function = lambda **kwargs: None
            self._routing_replay_out = None

    proposer.model.moe.router = CustomRoutingRouter()
    with pytest.raises(
        moe.DraftMoEConfigurationError,
        match="expected vllm.model_executor.layers.fused_moe.router",
    ):
        moe.validate_draft_moe(proposer)


def test_identity_accepts_the_multimodal_text_config_identifiers(proposer, monkeypatch):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    proposer.model.config = NS(
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=NS(model_type="gemma4_text"),
    )
    moe.validate_draft_moe(proposer)


def test_identity_rejects_a_gemma4_config_without_the_routing_contract(
    proposer, monkeypatch
):
    monkeypatch.setenv(moe.ENV_NAME, "4")
    del proposer.model.moe.router
    with pytest.raises(moe.DraftMoEConfigurationError, match="routing contract"):
        moe.validate_draft_moe(proposer)


def _topk4_proposer(proposer, monkeypatch, *, captured):
    """Build the production proposer and optionally run the real capture."""
    from unittest.mock import Mock

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.uno import UnoSpeculator

    monkeypatch.setenv(moe.ENV_NAME, "4")
    spec = object.__new__(UnoSpeculator)
    spec.k = 4
    spec.max_num_reqs = 2
    spec._step = 0
    spec.num_graph_replays = 0
    spec.num_eager_proposals = 0
    spec.num_warmup_proposals = 0
    spec.max_model_len = 64
    spec.model = proposer.model
    spec.device = torch.device("cuda")
    spec.vllm_config = proposer.vllm_config
    spec.speculative_config = NS(
        use_uno=lambda: True,
        uno_mask_token_id=1000,
        uno_noise_low=1,
        uno_noise_seed=42,
    )
    spec.lora_request = proposer.lora_request
    spec.input_buffers = InputBuffers(2, 8, torch.device("cpu"))
    spec.sample_idx_mapping = torch.empty(8, dtype=torch.int32)
    spec.draft_tokens = torch.empty((2, 4), dtype=torch.int64)
    spec.block_tables = NS(
        slot_mappings=torch.empty((1, 8), dtype=torch.int64),
        input_block_tables=[torch.ones((2, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    spec.kv_cache_config = Mock()
    spec.cudagraph_manager = Mock()
    spec.cudagraph_manager.cudagraph_mode = CUDAGraphMode.FULL
    spec.cudagraph_manager.graphs = []
    # A double must expose the manager surface the production coverage receipt
    # reads; a bare Mock hides that call instead of exercising it.
    spec.cudagraph_manager.captured_dispatch_keys.return_value = []
    spec.cudagraph_manager.key_is_covered.return_value = False
    spec.cudagraph_manager.dispatch_key.side_effect = (
        lambda num_tokens, num_active_loras: (num_tokens, num_active_loras)
    )
    spec._graph_attn_metadata = {}
    spec.set_lora_hook(lambda mapping: None)
    if captured:
        spec.capture()
    return spec


def _uncaptured_propose(spec, monkeypatch):
    """Arrange propose() to dispatch a descriptor with no captured graph."""
    from unittest.mock import Mock

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.spec_decode import uno as uno_module

    spec.cudagraph_manager.dispatch.return_value = BatchExecutionDescriptor(
        CUDAGraphMode.NONE, 8, 2
    )
    spec._copy_request_inputs = Mock()
    spec._build_uniform_attn_metadata = Mock(return_value={"eager": object()})
    spec._generate_draft = Mock()
    monkeypatch.setattr(uno_module, "prepare_uno_inputs_fused", Mock())
    monkeypatch.setattr(
        uno_module, "build_slot_mappings_by_layer", Mock(return_value={})
    )
    return NS(
        num_reqs=2,
        idx_mapping=torch.tensor([0, 1]),
        seq_lens_cpu_upper_bound=torch.tensor([10, 20]),
    )


def _propose(spec, batch, **kwargs):
    tensor = torch.empty(4)
    return spec.propose(
        batch,
        {},
        {},
        tensor,
        None,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        **kwargs,
    )


def test_uncaptured_serving_shape_is_refused_after_topk4_capture(
    proposer, monkeypatch, runtime_moe
):
    spec = _topk4_proposer(proposer, monkeypatch, captured=True)
    assert spec.draft_moe_state is not None
    assert spec.draft_moe_state.variant == "gemma4-sm86-marlin-topk4"
    assert proposer.model.moe.router.top_k == 8

    batch = _uncaptured_propose(spec, monkeypatch)
    with pytest.raises(runtime_moe.DraftMoEConfigurationError) as excinfo:
        _propose(spec, batch)
    message = str(excinfo.value)
    assert "gemma4-sm86-marlin-topk4" in message
    assert "num_reqs=2" in message
    assert "num_tokens=8" in message
    spec._generate_draft.assert_not_called()


def test_uncaptured_warmup_shape_stays_exempt(proposer, monkeypatch, runtime_moe):
    spec = _topk4_proposer(proposer, monkeypatch, captured=True)
    batch = _uncaptured_propose(spec, monkeypatch)
    _propose(spec, batch, dummy_run=True)
    spec._generate_draft.assert_called_once()


def test_uncaptured_profiling_shape_stays_exempt(proposer, monkeypatch, runtime_moe):
    """Both exemption arms must reach the guard, not just the dummy one."""
    spec = _topk4_proposer(proposer, monkeypatch, captured=True)
    batch = _uncaptured_propose(spec, monkeypatch)
    _propose(spec, batch, is_profile=True)
    spec._generate_draft.assert_called_once()


def test_uncaptured_refusal_names_the_same_key_as_the_coverage_receipt(
    proposer, monkeypatch, runtime_moe
):
    """The refusal must name the (num_tokens, effective_loras) key it lacked.

    The startup receipt states dispatch keys; a refusal that names only
    ``num_tokens`` leaves the reader unable to line the two up.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    spec = _topk4_proposer(proposer, monkeypatch, captured=True)
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 16, 4, num_active_loras=2)
    with pytest.raises(runtime_moe.DraftMoEConfigurationError) as excinfo:
        runtime_moe.refuse_uncaptured_eager_draft(spec, desc, warmup=False)
    message = str(excinfo.value)
    assert "num_tokens=16" in message
    assert "effective_loras=2" in message


def test_eager_serving_shape_is_refused_when_topk4_never_captured(
    proposer, monkeypatch, runtime_moe
):
    spec = _topk4_proposer(proposer, monkeypatch, captured=False)
    batch = _uncaptured_propose(spec, monkeypatch)
    with pytest.raises(
        runtime_moe.DraftMoEConfigurationError, match="requires captured"
    ):
        _propose(spec, batch)


def test_never_captured_refusal_names_key_environment_and_variant(
    proposer, monkeypatch, runtime_moe
):
    """The never-captured path must name all three identifiers.

    With no capture state there is no ``variant`` attribute to read, so this is
    the path that can drop the variant name while the captured-state path keeps
    it; the captured-state message is not a proxy for this one.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    spec = _topk4_proposer(proposer, monkeypatch, captured=False)
    assert getattr(spec, "draft_moe_state", None) is None
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, 16, 4, num_active_loras=2)
    with pytest.raises(runtime_moe.DraftMoEConfigurationError) as excinfo:
        runtime_moe.refuse_uncaptured_eager_draft(spec, desc, warmup=False)
    message = str(excinfo.value)
    assert f"{runtime_moe.ENV_NAME}={runtime_moe.DRAFT_TOP_K}" in message
    assert "num_tokens=16" in message
    assert "effective_loras=2" in message
    assert "gemma4-sm86-marlin-topk4" in message
