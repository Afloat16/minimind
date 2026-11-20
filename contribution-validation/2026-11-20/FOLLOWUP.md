# Submitted PR follow-up and pinned dependency checks

## Current submission state

All 20 planned changes have upstream PRs: **#927–#946**. The [status index](SUBMISSIONS.md) and [machine-readable snapshot](results/pr-status.json) record the exact check time, PR links, branches, heads, and GitHub mergeability. At that check, all 20 were open, non-draft, and cleanly mergeable, with no missing or duplicate submission. All 20 permit maintainer edits.

The current heads, titles, and complete descriptions match the prepared submission manifest. There are no discussion or inline-review comments to address, and no commit statuses, check runs, or PR workflow runs are reported. This is not a claim that an automatic CI suite passed. The upstream baseline remains `cc9629e5f464ef46ef9411bfac916364e0a6dff1`.

The current batch has **0 merged PRs**. The five previously merged contributions are outside this batch. Final acceptance and merging remain upstream maintainer actions.

## Additional compatibility evidence

The original verification environment used Pydantic 2.14.4 and Jinja2 3.1.6. The repository's [requirements.txt](https://github.com/jingyaogong/minimind/blob/cc9629e5f464ef46ef9411bfac916364e0a6dff1/requirements.txt) pins **Pydantic 2.11.5** and **Jinja2 3.1.2**. To check this specific dependency difference, six existing API, generation, and tokenizer regression classes were rerun with those pinned versions and matching pydantic-core 2.33.2.

The rerun used the exact canonical PR-head commits in the table below, in separate detached checkouts. All six baseline cases reproduced their expected behavioral failure, and all six fixed heads passed: **6 pairs, 12 observations, 12 expected outcomes**. The existing production branches and source files were preserved.

Actual runtime: **Python 3.12.14, PyTorch 2.6.0+cpu, Transformers 4.57.6, FastAPI 0.120.1, OpenAI SDK 1.59.6**. Dependencies were installed in an isolated directory; the pre-existing virtual environment was not changed.

This is a targeted Pydantic/Jinja2 compatibility check. It is not a reproduction of the full reference Python 3.10/Ubuntu/CUDA environment, a live SGLang run, or a GPU, memory, convergence, or model-quality benchmark.

## Reproduce an individual pair

First prepare the CPU environment and baseline/candidate checkouts using [README.md](README.md). Set the candidate to the exact head shown in the table. From the directory containing the existing regression scripts, add the two dependency overrides:

```bash
MINIMIND_PINNED_DEPS="$(pwd)/pinned-deps"
python -m pip install --target "$MINIMIND_PINNED_DEPS" pydantic==2.11.5 jinja2==3.1.2
export PYTHONPATH="$MINIMIND_PINNED_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run the corresponding command below with the candidate checkout path in `MINIMIND_CANDIDATE`. To reproduce the original problem, run it again with the baseline checkout path. A baseline behavioral assertion failure is expected; an import or dependency error does not establish reproduction.

| PR | Exact tested head | Candidate command | Baseline / candidate |
|---|---|---|---|
| [#930](https://github.com/jingyaogong/minimind/pull/930) | `6093668f948045535d2ccb3b4711de40093b8eec` | `python model_dataset_regressions.py --repo "$MINIMIND_CANDIDATE" --case agent-history-tool-call-json` | reproduced / passed |
| [#937](https://github.com/jingyaogong/minimind/pull/937) | `c7f395c5e71b77b035641c66cc76b1f01b9a25c6` | `python serving_regressions.py "$MINIMIND_CANDIDATE" budget` | reproduced / passed |
| [#939](https://github.com/jingyaogong/minimind/pull/939) | `df771bfee76f08c5d1e0d264d47c5388205897bb` | `python serving_regressions.py "$MINIMIND_CANDIDATE" finish` | reproduced / passed |
| [#941](https://github.com/jingyaogong/minimind/pull/941) | `2327805326f7964e61abe66236c98a51374eb9e8` | `python serving_regressions.py "$MINIMIND_CANDIDATE" arguments` | reproduced / passed |
| [#942](https://github.com/jingyaogong/minimind/pull/942) | `c38f380669795044ebc757e11f0257d3599b11bb` | `python serving_regressions.py "$MINIMIND_CANDIDATE" text_parts` | reproduced / passed |
| [#946](https://github.com/jingyaogong/minimind/pull/946) | `433cd5a94517a2b657827bbf7b1113340fd8d833` | `python root_checks.py "$MINIMIND_CANDIDATE" agent_eos` | reproduced / passed |

## Recorded evidence

- [Pinned compatibility result](results/pinned-compatibility.json): actual versions, commands, source hashes, exact tested commits, and all 12 outcomes.
- [Canonical checkout metadata](results/pinned-head-checkouts.json): PR heads and equality with the previously tested trees.
- [Raw output and installation logs](results/pinned-logs): original before/after traces and successful results.
- [Exact executed check sources and runner](results/pinned-check-sources): the source files matching the hashes in the result. The runner records the original workspace layout; the individual commands above use the portable entry points in this directory.

Original absolute paths in the raw logs identify the execution environment. The existing portable scripts expose the same selected regression cases. No new test cases or production changes were needed for this compatibility run.
