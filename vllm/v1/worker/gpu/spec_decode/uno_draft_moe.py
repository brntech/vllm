# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow, opt-in Gemma MoE top-k variant for Uno draft graph capture.

The process must be started with ``UNO_DRAFT_MOE_TOPK=4`` to enable this
candidate.  The context manager is entered only by the startup draft-graph
capture: it validates the loaded model/backend, changes the Gemma routers for
the captured draft graphs, and restores every router before the context exits.
The restored routers keep serving the target's own verification batches at the
configured top-k, so the variant is confined to the captured draft graphs.  A
serving shape without a captured graph is refused: an eager forward would
otherwise route at the configured top-k while the deployment believes the
reduced top-k is active.

This candidate deliberately refuses unknown model, hardware, backend, routing
replay, parallelism, or workspace arrangements.  It is an experiment guard,
not a general dynamic top-k API.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode

ENV_NAME = "UNO_DRAFT_MOE_TOPK"
BASE_TOP_K = 8
DRAFT_TOP_K = 4
SM86 = (8, 6)
GEMMA_NUM_EXPERTS = 128
_UNSET = object()
_PROCESS_TOP_K: object = _UNSET


class DraftMoEConfigurationError(ValueError):
    """The loaded process is outside this experiment's safe contract."""


def _requested_top_k() -> int | None:
    raw = os.environ.get(ENV_NAME)
    if raw is None or not raw.strip():
        requested = None
    elif raw.strip() == str(DRAFT_TOP_K):
        requested = DRAFT_TOP_K
    else:
        raise DraftMoEConfigurationError(
            f"{ENV_NAME} must be exactly {DRAFT_TOP_K} when set; got {raw!r}"
        )

    # Once an active candidate has been observed, do not permit a live process
    # to switch between graph variants.  A disabled process may still opt in
    # before its first capture, which keeps CPU fixtures composable.
    global _PROCESS_TOP_K
    if requested is not None:
        if _PROCESS_TOP_K is _UNSET:
            _PROCESS_TOP_K = requested
        elif requested != _PROCESS_TOP_K:
            raise DraftMoEConfigurationError(
                f"{ENV_NAME} is process-fixed at {_PROCESS_TOP_K}; "
                f"cannot switch to {requested}"
            )
    elif _PROCESS_TOP_K is not _UNSET:
        raise DraftMoEConfigurationError(
            f"{ENV_NAME} was enabled as {_PROCESS_TOP_K} and cannot be removed "
            "after capture setup"
        )
    return requested


def uno_draft_moe_enabled() -> bool:
    """Return whether the explicit top-k=4 candidate is requested."""
    return _requested_top_k() == DRAFT_TOP_K


def _reset_process_variant_for_tests() -> None:
    """Reset the process guard for isolated CPU fixtures."""
    global _PROCESS_TOP_K
    _PROCESS_TOP_K = _UNSET


# The Gemma 4 identifiers vLLM itself registers. The architecture strings are
# the registry entries in ``vllm/model_executor/models/registry.py``; the model
# types are the values mapped by
# ``vllm/transformers_utils/model_arch_config_convertor.py``. The guard accepts
# exactly these strings and never scans arbitrary config text. Identity of the
# loaded implementation is never decided by name: it is the class the model
# registry resolves the loaded configuration's architecture to (see
# ``_registered_implementation``).
GEMMA4_ARCHITECTURES = frozenset(
    {
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "Gemma4DSparkModel",
        "Gemma4MTPModel",
    }
)
GEMMA4_MODEL_TYPES = frozenset(
    {
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_unified_text",
        "gemma4_mtp",
        "gemma4_dspark",
        "gemma4_assistant",
        "gemma4_unified_assistant",
    }
)


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _model_owners(proposer: Any) -> Iterator[Any]:
    owners = [getattr(proposer, "model", None)]
    runner = getattr(proposer, "runner", None)
    owners.append(getattr(runner, "model", None))
    for owner in owners:
        if owner is not None:
            yield owner


def _config_candidates(owner: Any) -> Iterator[Any]:
    for attr in ("config", "hf_config"):
        config = getattr(owner, attr, None)
        if config is not None:
            yield config
    model_config = getattr(owner, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        yield hf_config


def _identity_configs(proposer: Any) -> Iterator[Any]:
    """Yield the typed configuration objects that carry model identity.

    Only the ``model_type`` and ``architectures`` fields of these objects are
    read; no arbitrary configuration text is inspected.
    """
    owners = list(_model_owners(proposer))
    runner = getattr(proposer, "runner", None)
    vllm_config = getattr(proposer, "vllm_config", None)
    if vllm_config is None:
        vllm_config = getattr(runner, "vllm_config", None)
    if vllm_config is not None:
        owners.append(vllm_config)
    seen: set[int] = set()
    for owner in owners:
        for config in _config_candidates(owner):
            if id(config) in seen:
                continue
            seen.add(id(config))
            yield config
            text_config = getattr(config, "text_config", None)
            if text_config is not None and id(text_config) not in seen:
                seen.add(id(text_config))
                yield text_config


def _exact_markers(config: Any) -> Iterator[str]:
    for attr in ("model_type", "architectures"):
        value = getattr(config, attr, None)
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple, set)):
            yield from (item for item in value if isinstance(item, str))


def _identity_model_config(proposer: Any) -> Any | None:
    """The model configuration the loader resolved the target class from."""
    candidates: list[Any] = [getattr(proposer, "model_config", None)]
    for holder in (proposer, getattr(proposer, "runner", None)):
        if holder is None:
            continue
        candidates.append(getattr(holder, "model_config", None))
        vllm_config = getattr(holder, "vllm_config", None)
        if vllm_config is not None:
            candidates.append(getattr(vllm_config, "model_config", None))
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _registered_implementation(proposer: Any) -> type:
    """Resolve the loaded architecture through vLLM's own model registry.

    The resolver is the one the loader used to turn ``architectures`` into a
    class, so this binds to the registered implementation rather than to a
    parallel table of names.
    """
    model_config = _identity_model_config(proposer)
    if model_config is None:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 cannot identify the loaded model "
            "configuration to resolve the registered implementation"
        )
    architectures = (
        getattr(getattr(model_config, "hf_config", None), "architectures", None) or []
    )
    if not architectures:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the loaded configuration to name the "
            "architectures the model registry resolves"
        )

    from vllm.model_executor.model_loader.utils import get_model_architecture

    try:
        model_cls, _ = get_model_architecture(model_config)
    except Exception as exc:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 cannot resolve the registered implementation "
            f"for architectures {list(architectures)!r}"
        ) from exc
    return model_cls


def _check_gemma4(proposer: Any) -> None:
    """Require the registered Gemma 4 implementation and its exact markers.

    Identity is bound to the class the model registry resolves the loaded
    configuration's architecture to, and the guard requires the loaded model to
    be that class (or a subclass). A same-named class is not identity, and an
    architecture the registry cannot resolve is refused rather than falling
    back to a name. Configuration text is never stringified, so an unrelated
    configuration field cannot qualify a different implementation.
    """
    registered = _registered_implementation(proposer)
    loaded = [type(owner) for owner in _model_owners(proposer)]
    if not any(issubclass(cls, registered) for cls in loaded):
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the loaded model to be the "
            f"registered Gemma 4 implementation {_class_path(registered)}; "
            "loaded " + ", ".join(sorted({_class_path(cls) for cls in loaded}))
        )
    markers = {
        marker
        for config in _identity_configs(proposer)
        for marker in _exact_markers(config)
    }
    if not (markers & GEMMA4_ARCHITECTURES or markers & GEMMA4_MODEL_TYPES):
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the exact Gemma 4 model_type or "
            "architectures identifier in the loaded configuration"
        )


def _check_device_sm86(proposer: Any) -> None:
    device = getattr(proposer, "device", None)
    if device is None:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires an explicit CUDA proposer device"
        )
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise DraftMoEConfigurationError(
            f"cannot resolve proposer device {device!r}"
        ) from exc
    if device.type != "cuda":
        raise DraftMoEConfigurationError(
            f"UNO_DRAFT_MOE_TOPK=4 requires CUDA SM86; got {device}"
        )
    if not torch.cuda.is_available():
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires an available CUDA device"
        )
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != SM86:
        raise DraftMoEConfigurationError(
            f"UNO_DRAFT_MOE_TOPK=4 requires SM86; got SM{capability[0]}{capability[1]}"
        )


def _check_parallelism(proposer: Any) -> None:
    configs = [getattr(proposer, "vllm_config", None)]
    runner = getattr(proposer, "runner", None)
    configs.append(getattr(runner, "vllm_config", None))
    parallel = None
    for config in configs:
        if config is None:
            continue
        parallel = getattr(config, "parallel_config", None)
        if parallel is not None:
            break
    if parallel is None:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 cannot identify parallel_config"
        )
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        try:
            value = getattr(parallel, name)
        except Exception as exc:
            raise DraftMoEConfigurationError(
                f"UNO_DRAFT_MOE_TOPK=4 cannot identify {name}"
            ) from exc
        if value != 1:
            raise DraftMoEConfigurationError(
                f"UNO_DRAFT_MOE_TOPK=4 requires {name}=1; got {value}"
            )


def _named_modules(proposer: Any) -> Iterator[tuple[str, Any]]:
    roots = [getattr(proposer, "model", None)]
    runner = getattr(proposer, "runner", None)
    roots.append(getattr(runner, "model", None))
    seen: set[int] = set()
    found = False
    for root in roots:
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        named = getattr(root, "named_modules", None)
        if not callable(named):
            continue
        found = True
        yield from named()
    if not found:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 cannot enumerate model modules"
        )


@dataclass(frozen=True)
class MoERouterBinding:
    name: str
    runner: Any
    router: Any
    backend: str
    prepare_finalize: str
    fixed_top_k: int


@dataclass
class DraftMoECaptureState:
    top_k: int
    bindings: tuple[MoERouterBinding, ...]
    _mutated: list[MoERouterBinding]

    @staticmethod
    def variant_name(top_k: int) -> str:
        """The variant name the capture receipt and every refusal quote."""
        return f"gemma4-sm86-marlin-topk{top_k}"

    @property
    def variant(self) -> str:
        return self.variant_name(self.top_k)

    def activate(self) -> None:
        if self.top_k != DRAFT_TOP_K:
            raise DraftMoEConfigurationError(
                f"unsupported draft MoE top-k {self.top_k}; expected {DRAFT_TOP_K}"
            )
        for binding in self.bindings:
            if int(getattr(binding.router, "top_k", -1)) != binding.fixed_top_k:
                raise DraftMoEConfigurationError(
                    f"router {binding.name!r} changed before draft capture"
                )
            binding.router.top_k = self.top_k
            self._mutated.append(binding)

    def restore(self) -> None:
        errors: list[Exception] = []
        for binding in reversed(self._mutated):
            try:
                binding.router.top_k = binding.fixed_top_k
            except Exception as exc:  # pragma: no cover - pathological setter
                errors.append(exc)
        self._mutated.clear()
        if errors:
            raise RuntimeError(
                "UNO_DRAFT_MOE_TOPK=4 could not restore all Gemma router states"
            ) from errors[0]

    def receipt(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "router": binding.name,
                "top_k": self.top_k,
                "backend": binding.backend,
                "prepare_finalize": binding.prepare_finalize,
                "configured_top_k": binding.fixed_top_k,
            }
            for binding in self.bindings
        )

    def log_capture_receipt(self, logger: Any) -> None:
        for item in self.receipt():
            logger.info(
                "UNO_DRAFT_MOE capture router=%s top_k=%d backend=%s "
                "prepare_finalize=%s configured_top_k=%d",
                item["router"],
                item["top_k"],
                item["backend"],
                item["prepare_finalize"],
                item["configured_top_k"],
            )


def _fixed_top_k(value: Any, label: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise DraftMoEConfigurationError(f"cannot identify {label} top-k") from exc
    if value != BASE_TOP_K:
        raise DraftMoEConfigurationError(
            f"{label} must remain configured at {BASE_TOP_K}; got {value}"
        )
    return value


def _moe_contract_classes() -> tuple[type, type, type]:
    """The loaded MoE contract classes, taken from their own modules."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter,
    )

    return CustomRoutingRouter, MarlinExperts, MoEPrepareAndFinalizeNoDPEPModular


def _bindings(proposer: Any) -> tuple[MoERouterBinding, ...]:
    result: list[MoERouterBinding] = []
    seen: set[int] = set()
    router_cls, experts_cls, prepare_cls = _moe_contract_classes()
    for name, module in _named_modules(proposer):
        router = getattr(module, "router", None)
        routed_experts = getattr(module, "routed_experts", None)
        if router is None or routed_experts is None or id(module) in seen:
            continue
        seen.add(id(module))

        if not isinstance(router, router_cls):
            raise DraftMoEConfigurationError(
                f"MoE router {name or type(module).__name__!r} is "
                f"{_class_path(type(router))}, expected {_class_path(router_cls)}"
            )
        if not callable(getattr(router, "custom_routing_function", None)):
            raise DraftMoEConfigurationError(
                f"MoE router {name or type(module).__name__!r} has no custom route"
            )
        fixed_top_k = _fixed_top_k(
            getattr(router, "top_k", None), f"router {name or type(module).__name__}"
        )

        quant_method = getattr(routed_experts, "quant_method", None)
        kernel = getattr(quant_method, "moe_kernel", None)
        fused_experts = getattr(kernel, "fused_experts", None)
        if not isinstance(fused_experts, experts_cls):
            backend = type(fused_experts).__name__ if fused_experts is not None else ""
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} backend "
                f"{backend or '<unknown>'!r} "
                f"is not {_class_path(experts_cls)}"
            )
        backend = experts_cls.__name__
        if bool(getattr(quant_method, "is_monolithic", True)):
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} is monolithic"
            )
        prepare = getattr(kernel, "prepare_finalize", None)
        if not isinstance(prepare, prepare_cls):
            prepare_name = type(prepare).__name__ if prepare is not None else ""
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} prepare/finalize "
                f"{prepare_name or '<unknown>'!r} is not "
                f"{_class_path(prepare_cls)}"
            )
        prepare_name = prepare_cls.__name__
        if not callable(getattr(fused_experts, "workspace_shapes", None)):
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} has no shape-driven workspace"
            )
        max_tokens_fn = getattr(prepare, "max_num_tokens_per_rank", None)
        if not callable(max_tokens_fn):
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} workspace contract is unknown"
            )
        try:
            max_tokens = max_tokens_fn()
        except Exception as exc:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} workspace contract is unknown"
            ) from exc
        if max_tokens is not None:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} has a fixed dispatch workspace"
            )

        config = getattr(module, "moe_config", None) or getattr(
            routed_experts, "moe_config", None
        )
        if config is None:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} has no FusedMoEConfig"
            )
        _fixed_top_k(
            getattr(config, "experts_per_token", None),
            f"MoE {name or type(module).__name__} config",
        )
        _fixed_top_k(
            getattr(routed_experts, "top_k", None),
            f"MoE {name or type(module).__name__} RoutedExperts",
        )
        if int(getattr(config, "num_experts", -1)) != GEMMA_NUM_EXPERTS:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} must expose "
                f"{GEMMA_NUM_EXPERTS} experts"
            )
        parallel = getattr(config, "moe_parallel_config", None)
        if parallel is None:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} has no parallel config"
            )
        for attr in ("tp_size", "pcp_size", "ep_size", "dp_size", "sp_size"):
            try:
                value = getattr(parallel, attr)
            except Exception as exc:
                raise DraftMoEConfigurationError(
                    f"MoE {name or type(module).__name__} cannot identify {attr}"
                ) from exc
            if value != 1:
                raise DraftMoEConfigurationError(
                    f"MoE {name or type(module).__name__} "
                    f"requires {attr}=1; got {value}"
                )
        for attr in ("use_ep", "use_all2all_kernels", "enable_eplb"):
            try:
                value = getattr(parallel, attr)
            except Exception as exc:
                raise DraftMoEConfigurationError(
                    f"MoE {name or type(module).__name__} cannot identify {attr}"
                ) from exc
            if bool(value):
                raise DraftMoEConfigurationError(
                    f"MoE {name or type(module).__name__} uses {attr}"
                )
        if getattr(parallel, "use_ep", None) is None:
            raise DraftMoEConfigurationError(
                f"MoE {name or type(module).__name__} has unknown expert dispatch state"
            )

        for owner, label in (
            (router, "router routing replay"),
            (kernel, "kernel routing replay"),
            (fused_experts, "expert routing replay"),
        ):
            for attr in (
                "_routing_replay_out",
                "routing_replay_capture_fn",
                "_routing_replay_buffer",
            ):
                if getattr(owner, attr, None) is not None:
                    raise DraftMoEConfigurationError(
                        f"MoE {name or type(module).__name__} has {label} state"
                    )

        label = name or getattr(module, "layer_name", None) or type(module).__name__
        result.append(
            MoERouterBinding(
                name=str(label),
                runner=module,
                router=router,
                backend=backend,
                prepare_finalize=prepare_name,
                fixed_top_k=fixed_top_k,
            )
        )
    if not result:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the Gemma MoE routing contract "
            "(MoERunner with CustomRoutingRouter and MarlinExperts); the "
            "loaded implementation exposes none"
        )
    return tuple(result)


def validate_draft_moe(proposer: Any, top_k: int = DRAFT_TOP_K) -> DraftMoECaptureState:
    """Validate the fixed experiment contract and snapshot router state."""
    if top_k != DRAFT_TOP_K:
        raise DraftMoEConfigurationError(f"only draft top-k={DRAFT_TOP_K} is supported")
    _check_gemma4(proposer)
    _check_device_sm86(proposer)
    _check_parallelism(proposer)
    sc = getattr(proposer, "speculative_config", None)
    use_uno = getattr(sc, "use_uno", None)
    if sc is None or not callable(use_uno) or not use_uno():
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the Uno speculator"
        )
    # The captured graphs bake the routed top-k into their kernel shapes, so
    # the variant needs a draft graph capture; an eager-only draft never
    # reaches the capture scope and must not silently serve top-k 8 rows.
    manager = getattr(proposer, "cudagraph_manager", None)
    mode = getattr(manager, "cudagraph_mode", None)
    if manager is None or mode is None or mode == CUDAGraphMode.NONE:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires captured Uno draft graphs"
        )
    if getattr(proposer, "lora_request", None) is None:
        raise DraftMoEConfigurationError(
            "UNO_DRAFT_MOE_TOPK=4 requires the Uno adapter routing"
        )
    bindings = _bindings(proposer)
    return DraftMoECaptureState(top_k=top_k, bindings=bindings, _mutated=[])


@contextmanager
def draft_moe_capture_scope(proposer: Any) -> Iterator[DraftMoECaptureState | None]:
    """Activate top-k=4 only while the private startup graph is captured."""
    requested = _requested_top_k()
    if requested is None:
        yield None
        return
    state = validate_draft_moe(proposer, requested)
    try:
        state.activate()
        yield state
    finally:
        state.restore()


def _uncaptured_refusal_message(
    desc: Any, shape: str, *, top_k: int, captured: bool
) -> str:
    """Name the dispatch key, the environment variable and the variant.

    ``captured`` separates a deployment whose capture ran and lost this key
    from one whose capture never created the variant state; both refusals must
    name the same three identifiers, so a reader can line the message up with
    the startup coverage receipt whichever path refused.
    """
    variant = DraftMoECaptureState.variant_name(top_k)
    if not captured:
        return (
            f"{ENV_NAME}={top_k} requires captured Uno draft graphs and never "
            f"created the {variant} capture state; the uncaptured serving shape "
            f"{shape} has none, and falling back to eager would route at the "
            "configured top-k"
        )
    return (
        f"{ENV_NAME}={top_k} captured the {variant} draft graphs; the "
        f"uncaptured serving shape {shape} has none to replay, and an eager "
        "forward would route at the configured top-k. Refusing to fall back; "
        f"add a cudagraph_capture_sizes entry covering {desc.num_tokens} draft "
        "rows or lower max_num_seqs"
    )


def refuse_uncaptured_eager_draft(proposer: Any, desc: Any, *, warmup: bool) -> None:
    """Refuse an uncaptured serving shape while the draft top-k variant is on.

    A captured draft graph bakes the reduced top-k into its kernel shapes, so a
    serving shape without a captured graph cannot run under this variant: an
    eager forward would route at the configured top-k instead, silently mixing
    the two routings. Warmup and profiling forwards discard their result and
    stay exempt, but a real proposal is refused before any forward.
    """
    if warmup:
        return
    state = getattr(proposer, "draft_moe_state", None)
    # The dispatch this desc came from resolved its effective-LoRA case, so the
    # key named here is the same (num_tokens, effective_loras) key the startup
    # coverage receipt reports as covered or uncovered.
    shape = (
        f"num_reqs={desc.num_reqs} num_tokens={desc.num_tokens} "
        f"effective_loras={desc.num_active_loras}"
    )
    if state is None:
        if not uno_draft_moe_enabled():
            return
        raise DraftMoEConfigurationError(
            _uncaptured_refusal_message(desc, shape, top_k=DRAFT_TOP_K, captured=False)
        )
    raise DraftMoEConfigurationError(
        _uncaptured_refusal_message(desc, shape, top_k=state.top_k, captured=True)
    )
