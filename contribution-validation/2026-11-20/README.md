# MiniMind contribution verification — 2026-11-20

All 20 contributions are now submitted as [PRs #927–#946](SUBMISSIONS.md). Start with the [review guide](REVIEW_GUIDE.md) for the observed failures, patch sizes, and direct evidence. See [the follow-up record](FOLLOWUP.md) for checks with the repository-pinned Pydantic and Jinja2 versions, and [the SGLang protocol note](SGLANG_PROTOCOL.md) for the adapter compatibility analysis.

This directory contains the regression checks used for 20 independent MiniMind contributions. Each command takes an explicit repository checkout and tests the matching fix branch. The common upstream baseline is:

```text
cc9629e5f464ef46ef9411bfac916364e0a6dff1
```

The checks use small CPU models, the repository's local tokenizer, temporary checkpoints, and deterministic fixtures. External inference services and generated responses are represented by fixtures where needed. The checks cover numerical behavior, gradients, input preservation, control flow, checkpoint contents, and tensor lifetimes. GPU performance, GPU peak memory, training convergence, and model-quality benchmarks were not run.

The checkpoint-staging check mocks the final `cuda:0` transfer while checking that checkpoint tensors are loaded on CPU and released before transfer. It does not require a GPU. The SGLang check uses local response fixtures and does not contact a SGLang service. The web checks execute the actual application function with UI fixtures; they do not require Streamlit or a browser.

## Files

| File | Cases |
|---|---:|
| `model_dataset_regressions.py` | 4 model/dataset cases |
| `training_checks.py` | 5 trainer/checkpoint cases |
| `serving_regressions.py` | 7 API, web, evaluation, and export cases |
| `root_checks.py` | 4 PPO, rollout, and agent cases |
| `requirements.txt` | Versions from the recorded verification environment |

The scripts are kept on the separate verification branch. They do not add a test framework or workflow to the production fix branches.

## Environment

The recorded runs used Python 3.12, PyTorch 2.6.0+cpu, and Transformers 4.57.6 on Linux. Run these commands from the directory containing this README and the four scripts:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
```

Installing dependencies and fetching Git branches need network access. The regression commands use local files and do not download pretrained weights or datasets.

## Prepare an explicit baseline and candidate

For example, to reproduce the text-content-parts fix:

```bash
git clone --no-checkout https://github.com/Afloat16/minimind.git review-repository
git -C review-repository fetch origin fix/api-text-content-parts
git -C review-repository worktree add --detach ../candidate FETCH_HEAD
git -C review-repository worktree add --detach ../baseline cc9629e5f464ef46ef9411bfac916364e0a6dff1

MINIMIND_CANDIDATE="$(cd candidate && pwd)"
MINIMIND_BASELINE="$(cd baseline && pwd)"

python serving_regressions.py "$MINIMIND_CANDIDATE" text_parts
python serving_regressions.py "$MINIMIND_BASELINE" text_parts
```

The candidate command should pass. The baseline command should fail an assertion showing that the rendered user message is empty. A dependency/import error is an environment failure and is not evidence that the regression reproduced.

For another contribution, create the candidate checkout from its branch in the table below and use the corresponding command. The branches are independent: running every case against a checkout containing only one fix will also exercise unrelated, unfixed problems. Existing checkouts can be used directly by assigning their absolute paths to `MINIMIND_CANDIDATE` and `MINIMIND_BASELINE`.

## Commands for all 20 contributions

### Model and dataset

| Branch | Command |
|---|---|
| `fix/decode-single-token-sdpa` | `python model_dataset_regressions.py --repo "$MINIMIND_CANDIDATE" --case decode-single-token-sdpa` |
| `fix/causal-loss-fp32-reduction` | `python model_dataset_regressions.py --repo "$MINIMIND_CANDIDATE" --case causal-loss-fp32-reduction` |
| `fix/moe-router-softmax-fp32` | `python model_dataset_regressions.py --repo "$MINIMIND_CANDIDATE" --case moe-router-softmax-fp32` |
| `fix/agent-history-tool-call-json` | `python model_dataset_regressions.py --repo "$MINIMIND_CANDIDATE" --case agent-history-tool-call-json` |

### Training and checkpoints

| Branch | Command |
|---|---|
| `fix/distillation-router-loss` | `python training_checks.py router "$MINIMIND_CANDIDATE"` |
| `fix/distillation-teacher-autocast` | `python training_checks.py teacher "$MINIMIND_CANDIDATE"` |
| `fix/load-checkpoint-on-cpu` | `python training_checks.py cpu "$MINIMIND_CANDIDATE"` |
| `fix/dpo-release-reference-logits` | `python training_checks.py reference "$MINIMIND_CANDIDATE" --baseline "$MINIMIND_BASELINE"` |
| `fix/checkpoint-update-boundaries` | `python training_checks.py checkpoint "$MINIMIND_CANDIDATE"` |

The `reference` case also compares the candidate's two-step parameter updates with the explicit baseline checkout. The baseline must be the unmodified commit above. The checkpoint case executes five trainer loops and their resume behavior, then evaluates the actual save predicate from the two RL trainers.

### Serving, tools, and export

| Branch | Command |
|---|---|
| `fix/web-tool-history` | `python serving_regressions.py "$MINIMIND_CANDIDATE" history` |
| `fix/web-generation-budget` | `python serving_regressions.py "$MINIMIND_CANDIDATE" budget` |
| `fix/tool-eval-budget` | `python serving_regressions.py "$MINIMIND_CANDIDATE" tool_budget` |
| `fix/api-length-finish` | `python serving_regressions.py "$MINIMIND_CANDIDATE" finish` |
| `fix/conversion-complete-base` | `python serving_regressions.py "$MINIMIND_CANDIDATE" conversion` |
| `fix/api-tool-arguments-json` | `python serving_regressions.py "$MINIMIND_CANDIDATE" arguments` |
| `fix/api-text-content-parts` | `python serving_regressions.py "$MINIMIND_CANDIDATE" text_parts` |

The API cases use real FastAPI requests, the local tokenizer, and real streaming decoders with deterministic model-output fixtures. The argument-encoding case also exercises the OpenAI SDK. The web-budget case runs native MiniMind generation with controlled logits to isolate the length-budget behavior. Export checks create and inspect temporary model files.

### PPO and agent rollouts

| Branch | Command |
|---|---|
| `fix/ppo-single-critic-norm` | `python root_checks.py "$MINIMIND_CANDIDATE" critic_norm` |
| `fix/ppo-stop-at-kl-limit` | `python root_checks.py "$MINIMIND_CANDIDATE" ppo_early_stop` |
| `fix/sglang-rollout-integrity` | `python root_checks.py "$MINIMIND_CANDIDATE" sglang_integrity` |
| `fix/agent-train-turn-endings` | `python root_checks.py "$MINIMIND_CANDIDATE" agent_eos` |

## Reading the results

A successful case exits with status 0 and prints a pass message or its measured fixture values. The model/dataset script uses standard `unittest` output. To reproduce the original defect, run the same case against `MINIMIND_BASELINE`; for the DPO reference case, also retain `--baseline "$MINIMIND_BASELINE"`.

A before/after result establishes the specific behavior checked by that fixture. The files are not a full training or deployment test suite. Recorded results and the contribution manifest supplied alongside the scripts identify the submitted revisions.

## Combined patch compatibility

Starting from `cc9629e5f464ef46ef9411bfac916364e0a6dff1`, all 20 submitted commits were automatically cherry-picked in the recorded contribution-list order, with zero conflicts and zero manual resolutions. The resulting tree passed Python AST parsing for all 18 changed Python files and `git diff --check`. This establishes patch compatibility and static validity for that baseline and order. The runtime regression results describe the individual contribution branches; a full runtime suite on the combined tree and GPU benchmarks were not run.

## Portability

The package does not depend on the original workspace location. All repository inputs are explicit, the model/dataset path is resolved before imports, and DPO's comparison checkout is passed through `--baseline`. Repository source is read explicitly as UTF-8, and temporary checkpoint files use the operating system's temporary directory. Only the 20 submitted contribution cases are exposed.
