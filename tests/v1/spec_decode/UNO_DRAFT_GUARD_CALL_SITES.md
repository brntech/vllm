# Uno draft-guard call sites

Every call of `execute_model`, `sample_tokens`, `_run_speculator_proposal` and
`propose(`, and every reader of `dummy_run` / `is_profile`, under
`vllm/v1/worker/` (both runner versions). Derived from a sweep of the tree at
revision `e37f4cff33`, the parent of the commit that carries it: the commit
whose subject is `docs(uno): inventory the draft-guard call sites and correct
two test comments`, which adds this file. An earlier copy of this inventory was
untracked, so this is the first committed copy; it is not evidence that
predates the changes it accompanies.

The sweep was re-run at revision `debaebb531` to add the reader sites an
earlier copy had omitted, so the selectors below span every hit of the sweep
except the occurrences named here as non-readers. The sweep matches the
identifiers rather than only their reads, so it also returns the
`maybe_dummy_run_with_lora` helper and the V1 call of it
(`lora_model_runner_mixin.py:253`, `gpu_model_runner.py:6105`), the V1
drafter's own `dummy_run` calls (`gpu_model_runner.py:4688,4765,6225`) and
prose in docstrings, comments and error text; none of them reads either flag,
so none carries a row.

Legend for the last column: can `UnoSpeculator.propose` reach
`refuse_uncaptured_eager_draft`
(`vllm/v1/worker/gpu/spec_decode/uno_draft_moe.py:638`) with **both** flags
false?

The refusal is reachable only from the eager branch of `UnoSpeculator.propose`
(`vllm/v1/worker/gpu/spec_decode/uno.py:1007`). `propose` has exactly two
callers, both in the V2 runner (`vllm/v1/worker/gpu/model_runner.py:944` and
`:1274`). `:944` hard-codes `dummy_run=True`. So the only path that can carry
both flags false is `:1274` (`_run_speculator_proposal`), and it carries
whatever the executing pass wrote into `ExecuteModelState` at
`gpu/model_runner.py:2332-2333`. Those values come from `execute_model`'s own
kwargs, which the worker supplies.

Row count: **A: 9 + B: 8 + C: 2 + D: 4 + E: 19 = 42 rows**, unchanged by the
re-sweep: the readers it added extend existing rows rather than adding rows.

## A. `execute_model` call sites

| # | site (file:line) | call | runner | flags passed | Uno refusal reachable with both false? |
| --- | --- | --- | --- | --- | --- |
| A1 | `vllm/v1/worker/worker_base.py:375` | `self.worker.execute_model(scheduler_output)` | both | none (`is_profile=False`, `dummy_run=False`) | yes, intended SERVING path (an uncaptured shape under the top-k variant is refused by design) |
| A2 | `vllm/v1/worker/gpu_worker.py:817-821` | `warmup_kernels(self.model_runner, self._discarded_startup_forward(), self.sample_tokens)` | V2 only (`gpu_worker.py:811`) | `is_profile=True` via the helper | no — exempt by `is_profile` |
| A3 | `vllm/v1/worker/gpu_worker.py:948-952` | `run_uno_served_jit_self_check(self.model_runner, self._discarded_startup_forward(), self.sample_tokens)` | both (V1 returns early: not a `UnoSpeculator`) | `is_profile=True` via the helper | no — exempt by `is_profile` |
| A4 | `vllm/v1/worker/gpu_worker.py:1246-1250` | `pass_state = {"is_profile": is_profile} if self.use_v2_model_runner else {}`, then `self.model_runner.execute_model(scheduler_output, intermediate_tensors, **pass_state)` | V1 + V2 shared | V2: the worker's `is_profile` (default `False`); V1: none | V2: yes, the serving transport; V1: no Uno speculator exists |
| A5 | `vllm/v1/worker/gpu/model_runner.py:889-897` | V2 `_dummy_run` calls `self.execute_model(..., dummy_run=True, skip_attn_for_dummy_run=skip_attn, is_profile=is_profile, ...)` | V2 | `dummy_run=True` | no — exempt by `dummy_run` |
| A6 | `vllm/v1/worker/gpu/warmup.py:458,461,463` (mixed prefill+decode) and `:811,871,909` (kernel warmup) | invokes the injected `worker_execute_model` | both | inherited from the callable handed in | no for the A2/A3 handlers (both are the marked helper); the other caller is A9 |
| A7 | `vllm/v1/worker/gpu_model_runner.py:4187` | V1 `def execute_model(self, scheduler_output, intermediate_tensors=None)` — no `is_profile`/`dummy_run` parameters | V1 | n/a | n/a — V1 has no Uno speculator |
| A8 | `vllm/v1/worker/mm_encoder_model_runner.py:94-102` | encoder-only `def execute_model(..., dummy_run=False, skip_attn_for_dummy_run=False, is_profile=False)`, asserts `not dummy_run` | encoder-only | n/a | no — encoder-only, no speculator |
| A9 | `vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py:169-172,183-186,275-278` | `run_mixed_prefill_decode_warmup(..., worker.execute_model, worker.sample_tokens, ...)` (outside `vllm/v1/worker/`) | V2, backend-gated | none | no — DeepSeek sparse-MLA / SM120 models only, never the Uno Gemma path |

## B. `sample_tokens` call sites

| # | site (file:line) | call | flags passed | note |
| --- | --- | --- | --- | --- |
| B1 | `vllm/v1/worker/worker_base.py:177-180` | abstract protocol | none | `sample_tokens` carries no flags; the pass classification lives in `execute_model`/`ExecuteModelState` |
| B2 | `vllm/v1/worker/gpu_worker.py:1176-1179` | `self.model_runner.sample_tokens(grammar_output)` | none | forwarding; V2 reads the classification from state (B8) |
| B3 | `vllm/v1/worker/gpu_worker.py:820` | `self.sample_tokens` handed to `warmup_kernels` (A2) | none | flags already on the execute side |
| B4 | `vllm/v1/worker/gpu_worker.py:951` | `self.sample_tokens` handed to the startup JIT self-check (A3) | none | flags already on the execute side |
| B5 | `vllm/v1/worker/gpu/warmup.py:459,462,829,872` | invokes the provided sample callable | none | mixed warmup + kernel warmup |
| B6 | `vllm/v1/worker/gpu/warmup.py:591-596` | `counted_sample_tokens` counts the call, then `return worker_sample_tokens(grammar_output)` | none | the counted forwarding the startup JIT self-check runs |
| B7 | `vllm/v1/worker/gpu_model_runner.py:4566` (V1) | `def sample_tokens`; reads the V1 `ExecuteModelState` at `:4591` | none | V1 state has no `dummy_run`/`is_profile` fields |
| B8 | `vllm/v1/worker/gpu/model_runner.py:2351-2372` (V2) | `def sample_tokens`; reads `dummy_run`/`is_profile` at `:2371-2372`; forwards at `:2536-2537` | from `ExecuteModelState` | the V2 transport a startup proposal depends on |

## C. `_run_speculator_proposal`

| # | site (file:line) | flags passed | note |
| --- | --- | --- | --- |
| C1 | `vllm/v1/worker/gpu/model_runner.py:1245-1258` | definition: `dummy_run=False, is_profile=False` parameters | |
| C2 | `vllm/v1/worker/gpu/model_runner.py:2526-2538` | `dummy_run=dummy_run, is_profile=is_profile` read from `ExecuteModelState` at `:2371-2372`, written by `execute_model` at `:2332-2333` | the ONLY call; the sole carrier of both-flags-false into `propose` |

## D. `propose(` call sites

| # | site (file:line) | flags passed | Uno refusal reachable? |
| --- | --- | --- | --- |
| D1 | `vllm/v1/worker/gpu/model_runner.py:944-965` (V2 `_dummy_run`) | `dummy_run=True, skip_attn_for_dummy_run=skip_attn, is_profile=is_profile` | no — exempt by `dummy_run` |
| D2 | `vllm/v1/worker/gpu/model_runner.py:1274-1290` (via C2) | `dummy_run=dummy_run, is_profile=is_profile` from the executing pass | yes iff that pass was serving (both false) and its shape is uncaptured |
| D3 | `vllm/v1/worker/gpu_model_runner.py:5042,5051,5076,5098,5125,5143,5277` (V1 `self.drafter.propose`) | no `dummy_run`/`is_profile` kwargs | no — V1 drafter, Uno V2 speculator absent |
| D4 | definitions: `gpu/spec_decode/speculator.py:67`, `autoregressive/speculator.py:204`, `dflash/speculator.py:301`, `extract_hidden_states.py:88`, `multi_module_mtp/speculator.py:132`, `uno.py:928` | signatures vary | only `uno.py:928` contains the guard |

## E. Readers of `dummy_run` / `is_profile` in `vllm/v1/worker/`

| # | site (file:line) | consumption | both-false consequence |
| --- | --- | --- | --- |
| E1 | `gpu_worker.py:973-987` (`_discarded_startup_forward`) | returns `partial(self.execute_model, is_profile=True)` | the mark that keeps A2/A3 out of the refusal, for both runner versions |
| E2 | `gpu_worker.py:1184` (parameter), `:1246` (keyword built), `:1248-1250` (forward) | worker→runner transport | the transport every pass shares |
| E3 | `gpu/model_runner.py:1961-1965` (V2 `execute_model` parameters), used at `:1995-1997,2010,2028,2032,2047,2081-2089,2090,2112,2135,2137,2144,2161,2170,2215,2313,2332-2333` | `dummy_run` gating of request state / inputs / attention; `is_profile` forces eager dispatch (`:2028`) and drops Mamba groups from profile attention (`:2144`); a non-first PP rank slices its persistent intermediate tensors on a dummy pass, where the serving pass copies the executing pass's own in (`:2215`) | a serving pass is (correctly) both false |
| E4 | `gpu/model_runner.py:1968-1981` | `if not dummy_run:` updates, adds, frees and applies request-state writes, then returns an empty output early when no token is scheduled (`:1976-1981`) | a dummy pass mutates no request state, and the early return precedes any proposal |
| E5 | `gpu/model_runner.py:1544-1559` (`gather_batch_req_state(..., dummy_run)`) | the `dummy_run` branch at `:1555-1559` returns uniform CPU state immediately | dummy passes never gather real per-request state |
| E6 | `gpu/model_runner.py:2712-2716` (`ExecuteModelState` fields) | state carried `execute_model`→`sample_tokens` | the values B8 and C2 read back |
| E7 | `gpu/model_runner.py:829-841,881-897,944-965` (V2 `_dummy_run`), callers `:1126-1128` (`profile_run`), `:1207` (draft-kernel warmup), `:1378` (adaptive-verification batches) | classifies its own pass (`dummy_run=True`) and forwards its `is_profile`; the memory profile passes `skip_attn=True, is_profile=True` | never both false |
| E8 | `gpu_worker.py:804,914-918,1353-1355` | `_dummy_run` callers: weight warmup, V1 sampler warmup, `execute_dummy_batch` | internal `dummy_run` classification; not a V2 proposal path |
| E9 | `gpu/eplb_utils.py:21-32,115-128` | `is_dummy`/`is_profile` EPLB gating | not a proposal path |
| E10 | `gpu/lora_utils.py:39-50` | `get_num_active_loras_for_dispatch(..., dummy_run)` | picks the effective-LoRA case dispatch resolves; not a proposal path |
| E11 | `gpu/spec_decode/autoregressive/speculator.py:226-229,301,336,367,385` | `is_profile`→eager (`:301,367`); `dummy_run`→skip the PCP block-table gather (`:336`) and, with `skip_attn_for_dummy_run`, the multi-step decode's attention (`:385`) | upstream speculator, not Uno |
| E12 | `gpu/spec_decode/dflash/speculator.py:323-326,348,369,414,442` | same pattern; `dummy_run`→skip the PCP block-table gather (`:369`) and the context-KV store, whose slots stay unset while the block tables are placeholders (`:414`) | upstream speculator, not Uno |
| E13 | `gpu/spec_decode/extract_hidden_states.py:102-105,114,116,120` | forwards both (`:114,116`); `skip_attn_for_dummy_run` gates the skip (`:120`) | upstream speculator, not Uno |
| E14 | `gpu/spec_decode/multi_module_mtp/speculator.py:154-157,201,209` | `is_profile`→eager (`:201`), `dummy_run`→skip (`:209`) | upstream speculator, not Uno |
| E15 | `gpu/spec_decode/speculator.py:89-92` | base signature | framework |
| E16 | `gpu/spec_decode/uno.py:942-945` (params), `:952,981,988,1001,1008-1009,1027` | `dummy_run and skip_attn_for_dummy_run` early-returns before the guard; `is_profile` forces an eager descriptor; `warmup=dummy_run or is_profile` exempts the guard | the guard under check |
| E17 | `gpu_model_runner.py:5850,5858,5963,6251,6523-6525,6951,6972` (V1 `_dummy_run` and `_warmup_and_capture`) | V1 dummy/profile classification; no proposal forwarding | V1 has no Uno speculator |
| E18 | `gpu_model_runner.py:3406-3417` (V1 `eplb_step`), `:4248` (empty DP batch) | `is_profile`; `self._dummy_run(1)` | not a proposal path |
| E19 | `mm_encoder_model_runner.py:76,98-103` | encoder-only | no Uno |

## What the fixes are checked against

1. **The shared worker forward.** A4/E2 is the only transport into
   `model_runner.execute_model`, and it is shared by V1 and V2. V1's
   `execute_model` (A7) has no `is_profile` parameter, so the keyword is sent
   only on the V2 branch, which leaves the V1 call contract byte-identical.
2. **The startup mark.** The only `propose` path that can arrive with both
   flags false from startup is D2 fed by A3/E1. A2/A5/D1 are already marked,
   and A3 uses the same helper as A2. V1 cannot reach it (Uno is V2-only and
   the self-check returns early), so the mark is safe for both runners. A
   serving proposal is still unmarked and still refused by name.
3. **The receipt and the refusal.** `dispatch`
   (`vllm/v1/worker/gpu/cudagraph_utils.py:570`) resolves through
   `resolve_dispatch` (`:512`), which applies padding (via `_candidates`), the
   effective-LoRA case and the `_graphs_captured` condition.
   `captured_dispatch_keys` (`:500`) enumerates that same padded key space and
   is empty before capture, `key_is_covered` (`:545`) calls the same
   resolution, and the startup receipt
   (`vllm/v1/worker/gpu/spec_decode/uno.py:840`) reports its keys, so the key a
   receipt advertises is the key dispatch consults and the key a refusal names.
